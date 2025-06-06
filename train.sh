exp_name_stage1=AniMerStage1
exp_name_stage2=AniMerStage2
experiment1=AniMerStage1
experiment2=AniMertage2
python main_amr.py exp_name=$exp_name_stage1 experiment=$experiment1 trainer=gpu launcher=local
cp -r ./logs/train/runs/$exp_name_stage1 ./logs/train/runs/$exp_name_stage2
python main_amr.py exp_name=$exp_name_stage2 experiment=$experiment2 trainer=gpu launcher=local