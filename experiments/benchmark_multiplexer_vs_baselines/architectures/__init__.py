"""4 BridgeArchitecture wrappers for Q1 benchmark.

- gtm: wraps track_p.multiplexer.GammaThetaMultiplexer
- recursive_link: 2-layer linear projection + cosine alignment
  (port of Yang et al. 2026 RecursiveMAS RecursiveLink)
- mlp_bridge: vanilla 2-layer MLP, same hidden_dim
- cross_attention: single-head cross-attention bridge
"""
