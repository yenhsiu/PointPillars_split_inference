import time
import torch
import numpy as np
import argparse
import pandas as pd
import os
from pointpillars.model.quantizations import RQBottleneck
import torch.nn.functional as F

def optimized_forward(self, x):
    """
    Optimized forward pass without slow sanity checks (CPU-GPU sync).
    Monkey-patched into RQBottleneck.
    """
    x_reshaped = self.to_code_shape(x)
    residual = x_reshaped.detach().clone()
    aggregated_quants = torch.zeros_like(x_reshaped)
    
    code_list = []
    vq_losses = []
    codebook_losses = []

    for codebook in self.codebooks:
        # Skip inactive codebooks
        if codebook.active_n_embed == 0:
            continue

        # Find nearest embedding indices for current residual
        with torch.no_grad():
            codes = codebook.find_nearest_embedding(residual)

        # REMOVED SLOW CHECKS HERE (no .item() calls)

        quant = codebook.embed(codes)

        # Update residual and accumulate quantized values
        if codebook.trainable:
            if codebook.use_ema and self.training:
                # EMA mode: Update codebook via EMA
                codebook._update_buffers(residual, codes)
                codebook._update_embedding()
                
                # Detach to prevent gradient flow in EMA mode
                residual = residual - quant.detach()
                aggregated_quants = aggregated_quants + quant.detach()
            else:
                # Gradient mode: Compute VQ losses with gradients
                vq_losses.append(F.mse_loss(residual.detach(), quant))
                codebook_losses.append(F.mse_loss(quant.detach(), residual))
                
                # Update residual and accumulate quantized values (keep gradients)
                residual = residual.detach() - quant.detach()
                aggregated_quants = aggregated_quants + quant
        else:
            # Frozen codebook: Detach everything
            residual = residual - quant.detach()
            aggregated_quants = aggregated_quants + quant.detach()

        code_list.append(codes.unsqueeze(-1))

    # Handle edge case: no active codebooks
    if not code_list:
        final_codes = torch.empty(*self.code_shape[:-1], 0, device=x.device, dtype=torch.long)
        zero_loss = torch.tensor(0.0, device=x.device)
        return x, zero_loss, zero_loss, final_codes

    # Concatenate codes from all codebooks
    final_codes = torch.cat(code_list, dim=-1)

    # Compute final losses
    if self.training and vq_losses:
        final_vq_loss = torch.mean(torch.stack(vq_losses))
        final_codebook_loss = torch.mean(torch.stack(codebook_losses))
    else:
        final_vq_loss = torch.tensor(0.0, device=x.device)
        final_codebook_loss = torch.tensor(0.0, device=x.device)

    # Reshape to original latent shape
    quants_final = self.to_latent_shape(aggregated_quants)

    return quants_final, final_vq_loss, final_codebook_loss, final_codes

# Apply monkey-patch
RQBottleneck.forward = optimized_forward

def create_rq_bottleneck(latent_shape, code_shape, codebook_size, decay=0.99, device='cuda'):
    rq = RQBottleneck(
        latent_shape=latent_shape,
        code_shape=code_shape,
        n_embed=codebook_size,
        decay=decay,
        ema=False,
        shared_codebook=False,
        restart_unused_codes=True,
        commitment_loss='cumsum'
    )
    return rq.to(device)

def measure_latency_for_emb(emb, max_codebooks=10, batch_size=1, iterations=100, device='cuda'):
    # Default spatial dimensions
    H, W = 496, 432
    latent_shape = (H, W, emb)
    # Create MAX model with max_codebooks
    code_shape = (H, W, max_codebooks)
    codebook_size = 64 # Default
    
    print(f"Creating Max Model: emb={emb}, max_codebooks={max_codebooks}")
    
    try:
        rq = create_rq_bottleneck(latent_shape, code_shape, codebook_size, device=device)
        rq.eval()
        
        dummy_features = torch.randn(batch_size, H, W, emb, device=device)
        
        # Warmup the whole model first
        print("  Warming up max model...")
        
        # Enable cudnn benchmark for potential speedup
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True
            
        with torch.no_grad():
             rq.set_evaluation_stage(max_codebooks, codebook_size)
             for _ in range(10):
                _ = rq(dummy_features)
        
        results = []
        
        for n_codebook in range(1, max_codebooks + 1):
            # Set active codebooks
            rq.set_evaluation_stage(num_codebooks=n_codebook, num_embeddings=codebook_size)
            
            # Use JIT Trace to optimize execution
            # This compiles the model for the specific active codebook configuration
            with torch.no_grad():
                traced_rq = torch.jit.trace(rq, dummy_features)
            
            # Small warmup for this specific configuration
            with torch.no_grad():
                for _ in range(5):
                    _ = traced_rq(dummy_features)
            
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                for _ in range(iterations):
                    _ = traced_rq(dummy_features)
            torch.cuda.synchronize()
            t1 = time.time()
            
            avg_latency = (t1 - t0) / iterations * 1000 # ms
            results.append({
                'emb': emb,
                'n_codebook': n_codebook,
                'latency_ms': avg_latency
            })
            print(f"  n_codebook={n_codebook}: {avg_latency:.4f} ms")
            
        return results

    except Exception as e:
        print(f"Error measuring emb={emb}: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    parser = argparse.ArgumentParser(description='Measure RQ Latency (Max Model)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--device', default='cuda', help='Device to run on')
    parser.add_argument('--output', default='latency_results/benchmark_pp_max_model.xlsx', help='Output file path')
    args = parser.parse_args()

    emb_sizes = [16, 32, 64]
    max_codebooks = 10
    
    all_results = []
    
    for emb in emb_sizes:
        emb_results = measure_latency_for_emb(emb, max_codebooks, args.batch_size, args.iterations, args.device)
        all_results.extend(emb_results)
    
    if not all_results:
        print("No results collected.")
        return

    df = pd.DataFrame(all_results)
    print("\nSummary:")
    print(df)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        df.to_excel(args.output, index=False)
        print(f"Results saved to {args.output}")
    except ImportError:
        print("openpyxl not installed, saving as csv")
        csv_output = args.output.replace('.xlsx', '.csv')
        df.to_csv(csv_output, index=False)
        print(f"Results saved to {csv_output}")

if __name__ == '__main__':
    main()
