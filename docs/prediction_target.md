# FinGraph AI — Prediction Target Specification

## Primary Prediction Task

The primary prediction task is five-trading-day directional prediction.

Given all information legitimately available at prediction time T, the system predicts whether the target security will have a positive or non-positive return over the following five trading sessions.

### Target Formula

```text
5D Return = (Price[T+5] - Price[T]) / Price[T]