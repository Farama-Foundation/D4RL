for e in $(cat scripts/reference_scores/envs.txt)
do
    python scripts/reference_scores/generate_ref_min_score.py --env_name=$e
done

