# Color-Object Association Task

This folder implements path patching experiments for a novel **Color-Object Association Task** using GPT-2 small, following the same mechanistic interpretability framework as the IOI task.

## Task Overview

### Task Structure
Given a sentence that introduces color-object pairs, the model must retrieve the correct object when given a preferred color.

**Example**:
```
"The red ball and blue cube are here. I prefer the red, so I'll take the"
→ Model should predict: "ball"
```

### Key Components
- **Color-Object Pairs**: "red ball", "blue cube"
- **Preference Statement**: "I prefer the red"
- **Task**: Model must map preferred color → associated object

---

## Expected Circuit Hypothesis

We hypothesize GPT-2 small implements the following algorithm:

### Human-Interpretable Algorithm:
1. **Bind** color-object pairs in context ("red" ↔ "ball", "blue" ↔ "cube")
2. **Identify** preferred color ("red" from "I prefer the red")
3. **Retrieve** object associated with preferred color ("ball")
4. **Output** retrieved object to logits

### Proposed Attention Head Classes:

#### 1. **Association Heads** (Early Layers 0-3)
- **Function**: Bind color-object pairs in context
- **Active at**: Color and object token positions
- **Attention pattern**: Color tokens attend to their paired objects
- **Signal**: Write association information to residual stream

#### 2. **Preference Detection Heads** (Mid Layers 4-7)
- **Function**: Identify which color is preferred
- **Active at**: END token position
- **Attention pattern**: Attend to color after "prefer the"
- **Signal**: Mark preferred color for retrieval

#### 3. **Color Retriever Heads** (Late Layers 8-10)
- **Function**: Look up object associated with preferred color
- **Active at**: END token position
- **Attention pattern**: Attend to object paired with preferred color
- **Signal**: Prepare object representation for output

#### 4. **Object Mover Heads** (Final Layers 10-11)
- **Function**: Output retrieved object to logits (analogous to Name Mover Heads)
- **Active at**: END token position
- **Attention pattern**: Attend to correct object token
- **Mechanism**: Copy object token via OV matrix to logits

### Expected Circuit Flow:
```
Input: "The red ball and blue cube are here. I prefer the red, so I'll take the"
         ↓
[Association Heads] → Bind "red"↔"ball", "blue"↔"cube"
         ↓
[Preference Detection] → Identify "red" as preferred color
         ↓
[Color Retriever] → Look up object paired with "red" → "ball"
         ↓
[Object Mover Heads] → Output "ball" to logits
         ↓
Prediction: "ball" (not "cube")
```

---

## Datasets

### Clean Dataset
- **Structure**: Proper color-object associations in context
- **Example**: `"The red ball and blue cube are here. I prefer the red, so I'll take the"`
- **Expected output**: "ball"
- **Logit difference**: `logit("ball") - logit("cube")` should be high

### Corrupt Dataset
- **Structure**: Preferred color NOT present in context
- **Example**: `"The green sphere and yellow pyramid are here. I prefer the red, so I'll take the"`
- **Expected output**: Model should be confused (no "red" object in context)
- **Logit difference**: Should be ~0 (model can't solve task)

### Dataset Parameters
- **Colors**: red, blue, green, yellow, purple, orange, pink, brown, black, white (10 colors)
- **Objects**: ball, cube, sphere, pyramid, cylinder, cone, prism, torus, disk, ring (10 objects)
- **Template**: `"The {color1} {object1} and {color2} {object2} are here. I prefer the {color1}, so I'll take the"`

---

## Path Patching Methodology

### Metric: Logit Difference
```python
logit_diff = logit(correct_object) - logit(incorrect_object)
```

- **Clean**: High logit_diff (model prefers correct object)
- **Corrupt**: Low logit_diff (~0, model confused)
- **Patched**: Recovery of logit_diff indicates head importance

### Direct Effect Path Patching (Object Movers)

Testing which heads directly affect logits (analogous to IOI Figure 3b):

**Algorithm**:
1. Run CLEAN forward pass → cache head h's output
2. Run CORRUPT forward pass → get corrupt logit_diff baseline
3. Run CORRUPT forward pass BUT replace head h with cached clean output
4. Measure: `patched_logit_diff - corrupt_logit_diff`

**Interpretation**:
- **Positive effect**: Head carries critical information (Object Mover candidate)
- **Negative effect**: Head works against task (Negative Object Mover)
- **Zero effect**: Head irrelevant for this task

