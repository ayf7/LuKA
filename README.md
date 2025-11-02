# LuKA — Learned Unimportant Key Aggregation

**A reversible, learnable approach to compressing attention KV-caches by aggregating *contiguous, low-importance* spans into surrogate keys (and re-expanding them on demand).** LuKA targets memory and latency in long-context inference while retaining the ability to “bring back” details when the model needs them.
