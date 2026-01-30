import os
from argparse import ArgumentParser



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--scene", type=str, default="skating", help="scene name")
    parser.add_argument("--save_dir", type=str, default="ckpt savedir", help="log path")
    parser.add_argument("--port", type=str, default="8811", help="log path")
    args = parser.parse_args()

    scene = args.scene
    save_dir = args.save_dir
    port = args.port

    # # # # # # # # 1: train camera motion predictor and static gaussians 
    os.system(f'python run_training_static.py --work-dir ./ckpts_high/{scene}/{save_dir} --port {port} data:stereohigh --data.data-dir ../dataset/stereo_high_dataset/{scene}')
    
    # # # # # # # # # # # 2: Load static gaussians, camera motion predictor, train dynamic gaussians, motion, and time parameters
    os.system(f'python run_training_dynamic.py --work-dir ./ckpts_high/{scene}/{save_dir} --port {port} data:stereohigh --data.data-dir ../dataset/stereo_high_dataset/{scene}')
    
    # # 3. Run testing and  compute metrics
    os.system(f'python run_testing.py --work-dir ./ckpts_high/{scene}/{save_dir} --port {port} data:stereohigh --data.data-dir ../dataset/stereo_high_dataset/{scene}')
    os.system(f'python run_compute_metrics.py --result_dir ./ckpts_high/{scene}/{save_dir} --gt_dir ../dataset/stereo_high_dataset/{scene}')
