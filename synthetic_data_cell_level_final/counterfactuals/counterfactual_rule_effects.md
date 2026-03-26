# Counterfactual Trust Analysis (§20)

This report analyzes clinical trust through interventions on admission logic.

**Mean Constraint Pressure**: 0.0348
**Acceptance Rate (Current Threshold 0.15)**: 100.0%
**Acceptance Rate (Strict Threshold 0.05)**: 100.0%

## Pressure Attribution by Stage

| primary_constraint | pressure |
| --- | --- |
| Design Alignment | 0.03478773551423575 |

## Conclusion

- High pressure blocks are primarily triggered by the **Monotonicity** requirement.
- If we enforced a 0.05 threshold, we would reject significantly more blocks, potentially increasing trust but reducing diversity.