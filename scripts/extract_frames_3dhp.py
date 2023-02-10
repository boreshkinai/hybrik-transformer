import os
import sys
import glob
from tqdm.auto import tqdm
root_folder = './mpi_inf_3dhp_train_set'

subjects = (1,2,3,4,5,6,7,8)
seqs = (1,2)
# fps = 25
frames = ((6416,12430),(6502,6081),(12488,12283),(6171,6675),(12820,12312),(6188,6145),(6239,6320),(6468,6054))
cmd = 'ffmpeg -i "<some_folder>/video_X.avi" -qscale:v 1 "<some_folder>/img_X_%06d.jpg"'

print("Converting videos to images...")

for si, s in enumerate(subjects):
    for sj, seq in enumerate(seqs):
        seq_parent_root = os.path.join(root_folder,'S'+str(s),'Seq'+str(seq))
        seq_root = os.path.join(seq_parent_root,'imageSequence')
        seq_output = os.path.join(seq_parent_root,'images')
        seq_videos = sorted(glob.glob(seq_root+'/video_*.avi'))
        if not os.path.exists(seq_output):
            os.mkdir(seq_output)
        else:
            os.system('rm -rf {}/*'.format(seq_output))
            
        pbar = tqdm(seq_videos)
        for v in pbar:
            i = int(os.path.basename(v).split('_')[-1][:-4])
            video_output = os.path.join(seq_output,'S{}_Seq{}_V{}'.format(s,seq,i))
            if not os.path.exists(video_output):
                os.mkdir(video_output)
            cmd = 'ffmpeg -v quiet -i "{}" -qscale:v 1 "{}/img_S{}_Seq{}_V{}_%06d.jpg"'.format(v,video_output,s,seq,i)
            pbar.set_description('S{} Seq{} V{}: {}/{}'.format(s,seq,i,len(os.listdir(video_output)),frames[si][sj]))
            os.system(cmd)
            assert len(os.listdir(video_output)) == frames[si][sj], 'Frames are less than gts. Aborted.'