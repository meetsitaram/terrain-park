# Training sets with terrain

Ready-to-train X2 clips from the Pico captures, each with the terrain it was captured on. Format: joblib pkl, clip name -> `root_trans_offset` (T,3), `root_rot` (T,4) xyzw, `dof` (T,31) MuJoCo order, `fps`.

| Set | Clips | Seconds | Terrain | Clips with terrain |
| --- | --- | --- | --- | --- |
| `acp_tp_ramp_sh2_x30` | 182 | 1647 | `ramp height map tp_ramp_hmap_x30.npz` | 182 |
| `pico_2026-10-29_2x2_test` | 3 | 14 | `pico_2026-10-29_2x2_props.json` | 3 |
| `pico_2026-10-29_2x2_train` | 10 | 49 | `pico_2026-10-29_2x2_props.json` | 10 |
| `tp_stairs_chain_x10` | 16 | 292 | `tp_stairs_chain_x10_props.json` | 16 |
| `tp_stairs_x10` | 8 | 45 | `tp_stairs_x10_props.json` | 8 |
| `tp_stairs_x10_sh2` | 88 | 491 | `tp_stairs_x10_sh2_props.json` | 88 |
| `tp_t4x4u_y20` | 56 | 217 | `tp_t4x4u_y20_props.json` | 56 |

`*_props.json`: clip name -> boxes (`center`, `half`, `yaw`) in the clip's world. Ramp clips use one shared height map (`x0`, `y0`, `res`, `grid`), so they have no boxes. `*_sh*` clips are the same takes with stride/height variations.
`acp_*_labels.json`: contact labels from the ACP work.
