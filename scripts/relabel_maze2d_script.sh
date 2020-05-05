for MAZE in open umaze medium large
do
    #python scripts/relabel_maze2d_rewards.py --maze=$MAZE --filename=maze2d-$MAZE.hdf5 --relabel_type=sparse
    #python scripts/relabel_maze2d_rewards.py --maze=$MAZE --filename=maze2d-$MAZE.hdf5 --relabel_type=dense
    python scripts/relabel_maze2d_rewards.py --maze=$MAZE --filename=maze2d-$MAZE-noisy.hdf5 --relabel_type=sparse
    python scripts/relabel_maze2d_rewards.py --maze=$MAZE --filename=maze2d-$MAZE-noisy.hdf5 --relabel_type=dense
done
