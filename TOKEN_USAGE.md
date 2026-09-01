## Total token usage

A basic vector-only RAG pipeline already existed. The tokens below were spent extending it
with hybrid search, reranking, and an MCP server.

```
 Session

   Total cost:            $49.96
   Total duration (API):  1h 15m 9s
   Total duration (wall): 1d 11h 4m
   Total code changes:    0 lines added, 0 lines removed
   Usage by model:
       claude-haiku-4-5:  1.3k input, 15 output, 0 cache read, 0 cache write ($0.0014)
          claude-opus-5:  6.6k input, 364.4k output, 65.9m cache read, 789.4k cache write ($49.96)
   Prompt cache (main):   203 requests · 99% of input tokens from cache · 1 miss (last 4h 56m 36s ago, 304.4k tokens re-cached) · warm (1h TTL, last activity 34m 10s ago)
   
```