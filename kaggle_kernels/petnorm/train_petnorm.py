"""
Kaggle kernel: PET-normalization data-engineering ablation
Attach dataset "adkspence/autopetv-ablation-subset" (private), enable GPU + internet.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nnunetv2==2.6.0", "SimpleITK==2.4.1"], check=True)

print("=== DIAGNOSTIC: /kaggle/input tree (depth 4) ===", flush=True)
for depth_root, dirs, files in os.walk("/kaggle/input"):
    depth = depth_root[len("/kaggle/input"):].count(os.sep)
    if depth > 3:
        dirs[:] = []
        continue
    print(f"  {depth_root} -> dirs={dirs} files={files[:5]}", flush=True)

candidates = list(Path("/kaggle/input").rglob("autopetv-ablation-subset"))
print(f"Candidates for dataset root: {candidates}", flush=True)
INPUT_DIR = candidates[0] if candidates else Path("/kaggle/input/autopetv-ablation-subset")
print(f"Using INPUT_DIR={INPUT_DIR}", flush=True)

WORK_DIR = Path("/kaggle/working")
DATASET_ID = 992
DATASET_NAME = f"Dataset{DATASET_ID}_AblationEdt"
CONFIGURATION = "3d_fullres"
FOLD = 0
NUM_EPOCHS = 90
TRAINER_NAME = f"nnUNetTrainer_ablation{NUM_EPOCHS}ep"

SPLIT = json.loads('''{
  "seed": 42,
  "num_per_tracer": 125,
  "train": [
    "fdg_01682f60c3_03112002",
    "fdg_06e7c24059_04222005",
    "fdg_07574bfa00_04202003",
    "fdg_08cdb15e0b_11212003",
    "fdg_099b3fd402_01032003",
    "fdg_0e2034240b_01312003",
    "fdg_0ea07b421b_10272001",
    "fdg_108c1763d4_09302004",
    "fdg_1253499c80_10072005",
    "fdg_1291700093_07132003",
    "fdg_13b40a817b_02182001",
    "fdg_147a9fcff3_08162001",
    "fdg_15a205ffcc_08192005",
    "fdg_15f4b7254f_10112003",
    "fdg_17d334cb6c_08282000",
    "fdg_18e8b02af3_08142005",
    "fdg_1956667fce_07282005",
    "fdg_1a1712f7d0_07052001",
    "fdg_1f6b6b0548_10162003",
    "fdg_25707f94a2_07032003",
    "fdg_29ab45ef17_06272002",
    "fdg_2b60c8135a_12092005",
    "fdg_30c4b7062b_01182001",
    "fdg_323cc5aff8_11092001",
    "fdg_35c9c85a96_09232005",
    "fdg_36870de2f2_05142006",
    "fdg_36d8219e3f_02092006",
    "fdg_3708f576ec_03082002",
    "fdg_3a4be713a1_11142002",
    "fdg_3b1c9155f5_01312003",
    "fdg_3b73c2480a_10192001",
    "fdg_3eac8f16d4_09302001",
    "fdg_4250de48c8_02122001",
    "fdg_42e9f16c09_01052003",
    "fdg_43323b7d42_05272001",
    "fdg_456d14846b_09012003",
    "fdg_49479d6e64_06072002",
    "fdg_4f7a8f41c0_05272001",
    "fdg_55ae7986e1_05102003",
    "fdg_581fa95eb0_03242003",
    "fdg_59e6d1de22_08062005",
    "fdg_5c55b3087d_12112000",
    "fdg_5d10be5b89_05302005",
    "fdg_5d6bf1e75f_12102000",
    "fdg_5e2da717db_06282002",
    "fdg_68b75093c5_11122000",
    "fdg_6a3477cd9a_01152005",
    "fdg_7323c415d0_02262006",
    "fdg_73fda3a382_08122005",
    "fdg_760c77b289_01202002",
    "fdg_76ebd5c736_12292005",
    "fdg_8de6953d23_01192001",
    "fdg_8e02f36295_11082004",
    "fdg_90ea6a6aaf_07272001",
    "fdg_92c5c944a5_07152002",
    "fdg_94986389d4_11112002",
    "fdg_9521502dbb_05172007",
    "fdg_97320b0b58_04172005",
    "fdg_983a76fd43_11012001",
    "fdg_9a583160ea_06212007",
    "fdg_9aa97cf103_07182004",
    "fdg_9d6699f215_06162003",
    "fdg_9da159b835_03232002",
    "fdg_9f7c68f5ca_12212001",
    "fdg_a1db71e797_03252005",
    "fdg_a4cd2b10de_12292001",
    "fdg_a86b3fad40_07182003",
    "fdg_ae96f738c0_05012003",
    "fdg_b1219c408b_10042002",
    "fdg_b258dfa7c2_12152002",
    "fdg_b3d4773f85_11042000",
    "fdg_b41bc7c1e5_03122005",
    "fdg_b6a1ee33ef_10242005",
    "fdg_b7c1533a39_02122005",
    "fdg_bbb83facb4_08232001",
    "fdg_c16c13211c_10262003",
    "fdg_c252d734a0_08092003",
    "fdg_c898e2abcb_12102004",
    "fdg_ca58410fad_05232003",
    "fdg_cbbc9e2879_01162006",
    "fdg_d4f3375362_05272005",
    "fdg_d51664a9e4_11052004",
    "fdg_d626611daf_11292002",
    "fdg_d63f6162a6_07082005",
    "fdg_d8d9e52cd5_05072007",
    "fdg_dc6174cb5d_03292003",
    "fdg_dc6174cb5d_12152002",
    "fdg_ddca6cfba6_02272004",
    "fdg_e4712dc58c_02121999",
    "fdg_e9be8ec30f_12122002",
    "fdg_e9e1a391b5_10112001",
    "fdg_ea6c621616_08082005",
    "fdg_ed9fa4eff1_05112001",
    "fdg_eeeda112bb_06072003",
    "fdg_f21755a99b_05052005",
    "fdg_f4668d2bdc_09042000",
    "fdg_f47e31ceb5_07262003",
    "fdg_f5c9b0de6c_04022004",
    "fdg_f7067b7bbb_05262001",
    "fdg_fb014a1ea0_12012005",
    "fdg_fe705ea1cc_12292002",
    "psma_0179419e313f7d8c_20190610",
    "psma_0198cdca94fbb95f_20191228",
    "psma_024e1c55ab49d220_20141004",
    "psma_05d59d060a8bb7d0_20200907",
    "psma_0d0f9fdf1578eaed_20220205",
    "psma_0ef9e2afd72f7483_20191004",
    "psma_1160f053324a19bd_20200117",
    "psma_12191f0bacb7e563_20160214",
    "psma_14c16d0ce2b3229d_20191019",
    "psma_18f866bfa3d793d4_20160801",
    "psma_19eed20451e33bc1_20190923",
    "psma_206e0f3bf8cb8ed3_20180309",
    "psma_2740fa8c813cd29c_20150102",
    "psma_27a4d77ee7e78fe3_20200606",
    "psma_28b47ab366f7ec9d_20160404",
    "psma_28f9ecc106933531_20170612",
    "psma_2d635a895be772d5_20190810",
    "psma_30fbb9471537bfb0_20140623",
    "psma_37328a98c70706f3_20201106",
    "psma_37af1d5c2373d0c4_20190930",
    "psma_41260c3678449a2f_20190503",
    "psma_41260c3678449a2f_20200612",
    "psma_41260c3678449a2f_20201010",
    "psma_438bb68482a3e054_20181013",
    "psma_44b93bfe059935a1_20140607",
    "psma_46cf1282a7648712_20170806",
    "psma_4da96443cf212c5f_20200831",
    "psma_4ee29f1dac1a4619_20200217",
    "psma_4f8549d7a1cb31b0_20160603",
    "psma_53c3a9d0f51745b6_20180519",
    "psma_576326cfba8460b8_20160212",
    "psma_602ab51cfa8fae3b_20140803",
    "psma_69ea0c011af2e2d4_20140921",
    "psma_6a3cdd0e7ed83b6c_20141018",
    "psma_6eeaaf6c168d8656_20150412",
    "psma_6ef5e5c442e4ce63_20180901",
    "psma_751d8b9d08e7859c_20220312",
    "psma_75b021ca6b8b39a0_20150104",
    "psma_799e088191559ae6_20171230",
    "psma_7b925af5d11d56e0_20170819",
    "psma_7bd096b4afea75d8_20180518",
    "psma_7e76841f3c9623c0_20160123",
    "psma_7ee9adc7b31f780e_20170409",
    "psma_7ee9adc7b31f780e_20190809",
    "psma_7fd33c03d8712a0b_20180825",
    "psma_80d66794f885f503_20191011",
    "psma_81b53d3911e09543_20201016",
    "psma_81b53d3911e09543_20210313",
    "psma_823a3a884418928b_20191223",
    "psma_834460a6d4330410_20200222",
    "psma_838c9340e85ca431_20150911",
    "psma_88c7a13c98fc9ede_20200427",
    "psma_8e7619b9167225fb_20160617",
    "psma_95b833d46f153cd2_20171118",
    "psma_9724be80190a0274_20160527",
    "psma_99278c16213e2429_20151219",
    "psma_9b7dfd431faf4067_20190727",
    "psma_9e37770ded7defaa_20150620",
    "psma_9ef8f0685c79f857_20160424",
    "psma_9ef8f0685c79f857_20190621",
    "psma_a4c539f7f753938b_20191004",
    "psma_a6509098ad09ada5_20160528",
    "psma_ab12d866a44fe389_20200904",
    "psma_b25df290eb7a867b_20190923",
    "psma_b3930f515d30fd6e_20191109",
    "psma_b3f377bd2b873391_20210807",
    "psma_b3f377bd2b873391_20211225",
    "psma_b5a4134e681683c0_20181005",
    "psma_b5a4134e681683c0_20181228",
    "psma_ba6433757dfc9d21_20211009",
    "psma_bc0699ac4dcbadee_20171105",
    "psma_bc70d4dbddba497e_20200925",
    "psma_bc70d4dbddba497e_20210130",
    "psma_be783bd2bf313946_20170505",
    "psma_be783bd2bf313946_20171028",
    "psma_bed76b3b7c2b172e_20200928",
    "psma_c0b18f90b401d6d6_20201214",
    "psma_c67186371fb31240_20190830",
    "psma_c76a99febb4bf130_20191025",
    "psma_c878220db97c05e1_20190902",
    "psma_ce956153f9464e26_20160521",
    "psma_d3e72b0003097756_20180924",
    "psma_d4b471bab61342ff_20200928",
    "psma_d5b636ea4da7638b_20190315",
    "psma_d7c4cf294221ea45_20190510",
    "psma_daf98725797ce51a_20160729",
    "psma_dc798b834a9962b6_20181029",
    "psma_dc8bb6a76a0a72cd_20161031",
    "psma_e21d5b67b48f750d_20141013",
    "psma_e2b34467658ec579_20170609",
    "psma_eb90cc5842f6109f_20181103",
    "psma_ee01cba36afe0f56_20180521",
    "psma_ee0d631fa63961f2_20160606",
    "psma_f21dfd4b31d6d5f2_20170107",
    "psma_f46740fcc3f44f10_20200522",
    "psma_f751ed4845fddae7_20170421",
    "psma_fad59ccacf4b88f2_20220326",
    "psma_fbd11b7e8c246d80_20181005",
    "psma_febfa344ff66b003_20181019"
  ],
  "val": [
    "fdg_21e4ffcb52_05062002",
    "fdg_27ad42f8a9_07142002",
    "fdg_37472e737f_06202003",
    "fdg_380f71df1e_08192001",
    "fdg_43647ff727_03022006",
    "fdg_442a09f90e_03162002",
    "fdg_4848bebb10_02232003",
    "fdg_53a0610615_07142001",
    "fdg_5d994c3f44_03052006",
    "fdg_605369e88d_09152002",
    "fdg_60baa6979c_08162003",
    "fdg_644d80e987_02152003",
    "fdg_80ccbdadf9_05062002",
    "fdg_a82f03863a_03192007",
    "fdg_ab953a5230_11032003",
    "fdg_ac11b344b6_11032002",
    "fdg_b510436d83_11102002",
    "fdg_b6fc20942c_12282001",
    "fdg_be3e55a32f_04052001",
    "fdg_d51bacdaba_06282001",
    "fdg_e252be4334_07092004",
    "fdg_ec581d49ef_03252002",
    "fdg_f2f28337ba_05032001",
    "fdg_fcdbe15200_12142003",
    "psma_19eed20451e33bc1_20200210",
    "psma_206e0f3bf8cb8ed3_20190419",
    "psma_2d1c74d01f72de21_20191104",
    "psma_310d8ea58fb29cb1_20200713",
    "psma_348a4dbda705c573_20170902",
    "psma_35650dda11e850e6_20180119",
    "psma_485138f63be9a0a4_20150110",
    "psma_4aa52e14d0947256_20150726",
    "psma_4c9d9614d81f3005_20191004",
    "psma_4dee75d0657f3c47_20151105",
    "psma_56a3a3e55545e167_20180629",
    "psma_7ba5d19cccb26c0e_20200210",
    "psma_8a5734c2659a5e0a_20200629",
    "psma_9c07b999c0e09e36_20200808",
    "psma_a2e474cdd4d3c64a_20191227",
    "psma_a63d490ca591ed15_20190209",
    "psma_d854b5837ca97440_20180204",
    "psma_dc21e961e5341e6b_20170630",
    "psma_dee1112c5d1380f1_20190225",
    "psma_e1eef1b5e130ded8_20160709",
    "psma_e88804f6e145e11a_20181202",
    "psma_e8c1c3c25d7a735b_20170730",
    "psma_ecbe2f11374632fa_20200926",
    "psma_f1888d6be1410fa4_20160502",
    "psma_f304b05dd35252f1_20191108",
    "psma_f8dd1ff555c3821f_20170915"
  ]
}''')
CASE_IDS = SPLIT["train"] + SPLIT["val"]

os.environ["nnUNet_raw"] = str(WORK_DIR / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(WORK_DIR / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(WORK_DIR / "nnUNet_results")

raw_dir = WORK_DIR / "nnUNet_raw" / DATASET_NAME
images_dir = raw_dir / "imagesTr"
labels_dir = raw_dir / "labelsTr"
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

print(f"Assembling nnU-Net raw dataset from {len(CASE_IDS)} cases...", flush=True)
import shutil
# Source files are named *.nii.gz.raw -- Kaggle auto-decompresses plain
# .gz files on upload (silently exploding them into folders), so they were
# renamed before upload to avoid that; rename back to .nii.gz here.
for case_id in CASE_IDS:
    shutil.copy(INPUT_DIR / "images" / f"{case_id}_0000.nii.gz.raw", images_dir / f"{case_id}_0000.nii.gz")
    shutil.copy(INPUT_DIR / "images" / f"{case_id}_0001.nii.gz.raw", images_dir / f"{case_id}_0001.nii.gz")
    shutil.copy(INPUT_DIR / "scribbles" / f"{case_id}_0002.nii.gz.raw", images_dir / f"{case_id}_0002.nii.gz")
    shutil.copy(INPUT_DIR / "scribbles" / f"{case_id}_0003.nii.gz.raw", images_dir / f"{case_id}_0003.nii.gz")
    shutil.copy(INPUT_DIR / "labels" / f"{case_id}.nii.gz.raw", labels_dir / f"{case_id}.nii.gz")

dataset_json = {
    "channel_names": {"0": "CT", "1": "PET", "2": "FG", "3": "BG"},
    "labels": {"background": 0, "lesion": 1},
    "numTraining": len(CASE_IDS),
    "file_ending": ".nii.gz",
}
(raw_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

preprocessed_dir = WORK_DIR / "nnUNet_preprocessed" / DATASET_NAME
preprocessed_dir.mkdir(parents=True, exist_ok=True)
splits = [{"train": SPLIT["train"], "val": SPLIT["val"]}] * 5
(preprocessed_dir / "splits_final.json").write_text(json.dumps(splits, indent=2))

import nnunetv2
trainer_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "training_length"
trainer_dir.mkdir(parents=True, exist_ok=True)
(trainer_dir / f"{TRAINER_NAME}.py").write_text(f'''
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class {TRAINER_NAME}(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = {NUM_EPOCHS}
        self.save_every = 10
''')


print("=== Planning (default planner) ===", flush=True)
subprocess.run(
    ["nnUNetv2_plan_and_preprocess", "-d", str(DATASET_ID), "--verify_dataset_integrity", "--no_pp"],
    check=True,
)

base_plans_path = preprocessed_dir / "nnUNetPlans.json"
new_plans_identifier = "nnUNetPlansPETClip"
new_plans_path = preprocessed_dir / f"{new_plans_identifier}.json"

plans = json.loads(base_plans_path.read_text())
plans["plans_name"] = new_plans_identifier
schemes = plans["configurations"][CONFIGURATION]["normalization_schemes"]
print(f"Original normalization_schemes: {schemes}", flush=True)
schemes[1] = "CTNormalization"
print(f"Modified normalization_schemes: {schemes}", flush=True)
new_plans_path.write_text(json.dumps(plans, indent=2))

print("=== Preprocessing (PET-clip plans) ===", flush=True)
subprocess.run(
    ["nnUNetv2_preprocess", "-d", str(DATASET_ID), "-plans_name", new_plans_identifier,
     "-c", CONFIGURATION, "-np", "2"],
    check=True,
)

print("=== Training (PET-norm) ===", flush=True)
subprocess.run(
    ["nnUNetv2_train", str(DATASET_ID), CONFIGURATION, str(FOLD),
     "-tr", TRAINER_NAME, "-p", new_plans_identifier],
    check=True,
)


print("Kernel run complete.", flush=True)
