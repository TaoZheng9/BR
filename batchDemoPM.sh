prefix="PMO"
cpdict="YOUR_CHECKPOINT_PATH"
cpvfile="visual_best.pth" # 
cpafile="audio_best.pth" # 
#cpvfile="visual_latest.pth"
#cpafile="audio_latest.pth"
datapath="YOUR_DATA_PATH"
start_vid=YOUR_START_VIDEO_ID
num_vid=GENERATION_NUM
outdir="./YOUR_OUTPUT/${prefix}/"

python xdemoPMO.py --unet_input_nc 2 --unet_output_nc 2 --prefix ${prefix} --start_vid ${start_vid} --num_vid ${num_vid} --input_audio_path ${datapath}binaural_audios --video_frame_path ${datapath}frames/ --weights_visual ${cpdict}/mono2binaural/${cpvfile} --weights_audio ${cpdict}/mono2binaural/${cpafile} --output_dir_root ${outdir} --input_audio_length 10 --hop_size 0.05