# T060-A spatial-exposure direction audit

Classification: `spatial-exposure projection is insufficiently active`. Nondegenerate65/80 fails fixed72/80 gate. All80 predicted gradients are nonzero; degeneracy is15 exactlyzero source-reference gradients, not a predicted activation failure. Retain the prescribed classification and cohort.

Authorization522cce475dd0ec82c5eeb81f8205e468068d6a70; source8d4a331536a9f6c910b7bde2bd8db486e7a11ba7; Uevidence88de78b1a91adfc6dec71c378cdf8a4ec044cb51. Exact80persisted Ustate0 anchors/16outer sourceimages/order. CohortSHAf3fabdec3e9397a133347094a5cd619153f50c9b5cd03e4348980b5a37534fd4. No cohort substitution or condition-based selection.

T051code pinned to PR76 head78d24ef2365ec2a8b3e18036a9154ec9f833e9d9 and sourceSHA0a75c8b083239bc5b4c242cdd83e2e19abd75f6f14771033cfb5584594c8cbd7. AST extracts exact ev() plus first z assignment: 2*tanh(u) on8x8 grid, bilinear align_cornersFalse, RGB-shared base*exp2(e),clamp0to1. No T051tone/oraclemodel import, trajectory, state or per-image selector. Complete current degradedbase exposure per explicit task; inherited Region2 state/grid and gatefeatureconstants unchanged. Each y0/grid reproduces U hashes.

Frozen Ehead e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0 and trainnormalization, CLIP/prototypes, original28-D features. Fresh target-free Jexp8 differentiable feature rows,20 fixed rows, q and J-transpose-q. No cachedmatched-detailJ/g used. Onlyu differentiated;u stayszero; optimizersteps0.

## Results

65 nondegenerate under bothnorms>1e-12;60/65positive dots=0.9230769230769231. Cosine mean0.33672857889293484,median0.3524152040016726,p10=0.03010917947689727,p90=0.664910713948927. Primary coveragegate fails. Descriptively,positivefraction exceeds.80 butmediancosine also falls below.40; no later-stage pass is claimed.

Zero-reference banks: [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 325, 350, 375]. The smallest included reference norm is4.306956464587262e-12;threshold unchanged. Full80 norm/dot/cosine/positive/nondegenerate rows committed, including all degenerate anchors. All80 and nondegenerate norm summaries plus dot summaries in result.json.

{
  "norms": {
    "predicted_norm": {
      "all80": {
        "mean": 0.7229304603394577,
        "median": 0.427568207333769,
        "p10": 0.15422589511280327,
        "p90": 1.891980185615172,
        "min": 0.04909897264046589,
        "max": 4.381889248791637
      },
      "nondegenerate": {
        "mean": 0.8387012324333128,
        "median": 0.5919118803531312,
        "p10": 0.1834349948743794,
        "p90": 1.9934047537069972,
        "min": 0.04909897264046589,
        "max": 4.381889248791637
      }
    },
    "reference_norm": {
      "all80": {
        "mean": 0.03098269542014327,
        "median": 0.029012014339447695,
        "p10": 0.0,
        "p90": 0.058262911808472866,
        "min": 0.0,
        "max": 0.07888438618242379
      },
      "nondegenerate": {
        "mean": 0.03813254820940709,
        "median": 0.036312103670814715,
        "p10": 0.016468052050992995,
        "p90": 0.05999209333264794,
        "min": 4.306956464587262e-12,
        "max": 0.07888438618242379
      }
    }
  },
  "dot": {
    "mean": 0.011731411739505777,
    "median": 0.004928356790320566,
    "p10": 0.0004584558452983874,
    "p90": 0.02899365648322573,
    "min": -0.003472329826666566,
    "max": 0.08203582147813289
  }
}

## Isolation, execution and validation

Target-free start 2026-09-18T22:23:53.295555+00:00;global80freeze 2026-09-18T22:24:17.955190+00:00;first sourcecleanread 2026-09-18T22:24:23.538292+00:00. All80x/Jexp/q/g saved/fsynced/hashed before sourceclean image or restorationgradient access. Filesystem audit restricts project access to pinnedcode/model/target-free bankinputs and outputdir; separate diagnostic opens16authorized sourcecleans, then independent verifier reopens same16. No reference value enters predicted field. All234sourcebindings,inputhashes,head and target-free outputs immutable.

Counters beforefreeze clean/reference0,referencegradient0,targetdomain0,LOL-v2 0,officialtest0,inferenceleakage0;optimizersteps0 throughout. MSE used only to differentiate source restoration at u=0;no finite-step MSE/PSNR/SSIM comparison or search. A6000GPU1 computes CLIP/J and referencegradients; CPUindependent analyticadjoint/scalar verification.

4tests pass3.49s. Independent80PASS reconstructs gR via closed-form fullRGB MSE/exposure derivative and bilinear adjoint, NumPyJ-transpose-q,scalar norms/dots/cosines/quantiles/counts/classification and immutable hashes. Maximum analytic gradient relative error 2.050581961487991e-06, maximum absolute error 4.529787095940152e-08. Fixed verifier thresholds declared before execution.

Sole run20260919-062347-ttie-t060a-exposure,exit0 at22:24:33UTC. No execution failure,repair or rerun. One SSH connection closed during preparation before files were created;recovered before experiment. Missing historical root-state lookup resolved using pinned GitHub PR source. Dfull/NVMLwarning remain nonblocking.

ResultSHA 0a2162b45f5468402e9c0f8baabbf35dff8c703b2d92f571744996ff9faad495. Raw archive {"home": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t060a/T060A_evidence.tar.gz", "backup": "/media/wenchang/F/wjq/TTIE/shared/t060a/T060A_evidence.tar.gz", "sha256": "1f9ec8ca0c6e9448befd7c23d45bbd37cc3915172851e65469dfb6e097d3e58f", "bytes": 101955506, "home_verified": true, "F_backup_verified": true}. Fullbaseimages+fields remotehome/F;compactfields/referencegradients/tables/report in Git. Next: stop under fixedcoverage gate and await review;do not change eligibility,drop zero-reference anchors,retrain,changegrid/range,addoptimizerstep or roll out to real domain.
