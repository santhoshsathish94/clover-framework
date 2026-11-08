# Context File — Paper 11: Ivaldi et al. (2014) — *Object Learning Through Active Exploration*

- **Citation**: Serena Ivaldi, Sao Mai Nguyen, Natalia Lyubova, Alain Droniou, Vincent Padois, David Filliat, Pierre-Yves Oudeyer & Olivier Sigaud (2014). *Object Learning Through Active Exploration*. **IEEE Transactions on Autonomous Mental Development**, Vol. 6, No. 1, pp. 56–72 (`doi:10.1109/TAMD.2013.2280614`; `hal-00919694`).
- **Source PDF / Extracted Text**:
  - Primary PDF URL: `https://hal.science/hal-00919694/file/Ivaldi_IEEE_TAMD_2013.pdf`
  - Extracted Full Text (`100,829 bytes`): [research/curiosity-and-developmental-learning/papers_text/11-ivaldi-2014-object-learning.txt](../../../research/curiosity-and-developmental-learning/papers_text/11-ivaldi-2014-object-learning.txt)
- **Executable Experiment & Saved Results**:
  - Experiment Script: [research/curiosity-and-developmental-learning/11-ivaldi-2014-object-learning/run_ivaldi2014_experiments.py](../../../research/curiosity-and-developmental-learning/11-ivaldi-2014-object-learning/run_ivaldi2014_experiments.py)
  - Saved Results Artifact: [research/curiosity-and-developmental-learning/11-ivaldi-2014-object-learning/results/ivaldi_2014_results.json](../../../research/curiosity-and-developmental-learning/11-ivaldi-2014-object-learning/results/ivaldi_2014_results.json)

---

## Stage 1 — Context (Primary Source & Cognitive Architecture)

### 1.1 The `MACSi` Cognitive Architecture on the 53-DOF `iCub` Humanoid (Section II, Figures 1–5)
Ivaldi et al. (2014) operationalize Smith & Gasser's (2005) *"Six Lessons from Infant Development"* (multimodality, incremental development, physical interaction, exploration, social guidance, and symbol grounding) in the `MACSi` architecture running across 20 concurrent YARP/ROS modules on the child-sized `iCub` humanoid (41 upper-body DOFs):
1. **Unsupervised Incremental Visual Hierarchy (Section II-D, Figures 3–4)**:
   - Segments moving regions via optical flow + depth contours into **proto-objects**, extracts `SURF` keypoints and `HSV` superpixel color features into pairs/triples (**mid-features**), quantizes them into incremental Bag-of-Words **views** via `TF-IDF` voting, and groups views into **entities**.
   - **Self / Human / Object Categorization (Figure 4)**: Uses mutual information between visual motion and robot arm proprioception to distinguish the **robot hand (`r`)**, the **human caregiver hand (`h`)**, **static/manipulable objects (`o`)**, and **grasped connected entities (`r+o`)**.
2. **`SGIM-ACTS` (Socially Guided Intrinsic Motivation with Active Choice of Teacher and Strategy, Section II-B & III-B)**:
   - At each decision step, the robot actively selects the triple `(object, action, actor)` — choosing which object to explore and whether to execute an autonomous manipulation (`push` or `lift/throw`) or request a social demonstration (`show/manipulate`) from the human caregiver.

---

## Stage 2 — Direction (Testable Hypotheses)

1. **Experiment E1 (Section III-A, Figures 8, 9 & 13 — Active Manipulation on 14 iCub Toys & Emergent `Rollable` Affordances)**:
   - Simulate 30 push-left trials on each of the 14 objects from Figure 8 (`gray_dog`, `green_box`, `blue_soda_can`, `blue_ball`, `red_bear`, `blue_octopus`, `white_bottle`, `blue_violet_sphere`, `chicken`, `red_soda_can`, `yellow_train`, `cow`, `yellow_car`, `patchwork_4_cubes`).
   - Verify Figure 9b (complex multi-faceted objects like `patchwork_4_cubes`, `yellow_train`, `chicken` yield $> 55\%$ view changes and 10–14 distinct views, whereas uniform `red_bear` yields $< 5\%$) and Figure 13 (two $y$-displacement thresholds at `0.15 m` and `0.30 m` cleanly separate `spheres`, `cylinders_or_wheels`, and `other` non-rolling objects).
2. **Experiment E2 (Section III-B, Figures 10, 11 & 12 — `SGIM-ACTS` vs. `Random` Under "Good/Unbiased" and "Bad/Biased" Human Teachers)**:
   - Evaluate `SGIM-ACTS` vs. `Random` on the 5-object, 64-image recognition benchmark (`car`, `cube`, `ball`, `bear`, `dog`) with an **Unbiased Teacher** (flips/rotates objects to reveal new views) and a **Biased Teacher** (refuses to rotate objects and always presents the same side).
   - Verify all three findings reported on p. 10: (1) `SGIM-ACTS` outperforms `Random` under both teachers, (2) `SGIM-ACTS` focuses $> 50\%$ of its time on the hardest object (`cube`, the 4-color patchwork), and (3) under the Biased Teacher, `SGIM-ACTS` compensates by shifting from `human_show` to high-perturbation `robot_lift_throw`.

---

## Stage 3 — Execution & Stage 4 — Outcome (Quantitative Results)

Running [run_ivaldi2014_experiments.py](../../../research/curiosity-and-developmental-learning/11-ivaldi-2014-object-learning/run_ivaldi2014_experiments.py) produces:

### 4.1 Experiment E1: 14-Object Push-Left View Discovery & Rolling Affordance Emergence
- **View-Change Extremes (matching Figure 9b, p. 9)**:
  - `5:red_bear`: `0.0%` view change (`1` view collected), `median_dy = -0.115 m` (`other`).
  - `14:patchwork_4_cubes`: **`76.7%`** view change (`14` views collected), `median_dy = -0.098 m` (`other`).
  - `11:yellow_train`: **`70.0%`** view change (`13` views collected), `median_dy = -0.255 m` (`cylinder_or_wheel`).
  - `4:blue_ball` & `8:blue_violet_sphere`: `median_dy = -0.518 m` and `-0.353 m` (`sphere`, $|dy| > 0.30\text{ m}$).
- **Figure 13 Rolling Affordance Classification Accuracy**: **`100.0%` (`14 / 14` objects)**.

### 4.2 Experiment E2: `SGIM-ACTS` vs. `Random` Under Unbiased and Biased Teachers (`64`-Image Database)
| Condition | Recognized Images (`/ 64`) | `cube` Share | `robot_push` | `robot_lift_throw` | `human_show` |
|---|---|---|---|---|---|
| **`Random` + Unbiased Teacher** | `55.07 / 64` | `21.53%` | `32.78%` | `33.89%` | `33.33%` |
| **`Random` + Biased Teacher** | `51.62 / 64` | `21.53%` | `32.78%` | `33.89%` | `33.33%` |
| **`SGIM-ACTS` + Unbiased ("Good") Teacher** | **`62.67 / 64`** | **`76.67%`** | `12.36%` | `50.28%` | **`37.36%`** |
| **`SGIM-ACTS` + Biased ("Bad") Teacher** | **`62.48 / 64`** | **`74.58%`** | `14.03%` | **`79.03%`** | **`6.94%`** |

---

## Stage 5 — Growth (What We Learned & Critical Boundaries)

1. **How Intrinsic Motivation Protects Social Learning Against Naive or Biased Human Partners**:
   - When the human partner is a "bad" (biased) teacher who repeatedly presents the same static view (`human_show` yields near-zero progress), `Random` exploration drops from `55.07` to `51.62 / 64` recognized views.
   - By contrast, `SGIM-ACTS` detects that `human_show` is yielding zero competence progress, cuts `human_show` requests from **`37.36%` down to `6.94%`**, and compensates by increasing its own high-perturbation `robot_lift_throw` manipulation from **`50.28%` to `79.03%`**, achieving **`62.48 / 64`** recognition despite the unhelpful teacher!
