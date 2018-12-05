ffmpeg -i .//mmnist/MMNIST_FutureGAN.gif -i .//kth/KTH_FutureGAN.gif -i .//cityscapes/Cityscapes_FutureGAN.gif -filter_complex \
"[0:v]pad=iw+10:ih+64:x=0:y=32:color=white[0]; \
 [1:v]pad=iw+10:ih:color=white[1]; \
 [2:v]pad=iw:ih:color=white[2]; \
 [0][1][2]hstack=3" \
Examples_FutureGAN.gif

ffmpeg -i .//mmnist/sequence0001_FutureGAN.gif -i .//mmnist/sequence0002_FutureGAN.gif -i .//mmnist/sequence0003_FutureGAN.gif -i .//mmnist/sequence0004_FutureGAN.gif -i .//mmnist/sequence0005_FutureGAN.gif -i .//mmnist/sequence0006_FutureGAN.gif -i .//mmnist/sequence0001_GroundTruth.gif -i .//mmnist/sequence0002_GroundTruth.gif -i .//mmnist/sequence0003_GroundTruth.gif -i .//mmnist/sequence0004_GroundTruth.gif -i .//mmnist/sequence0005_GroundTruth.gif -i .//mmnist/sequence0006_GroundTruth.gif -filter_complex \
"[0:v]pad=iw+30:ih+10:color=white[t0]; \
 [1:v]pad=iw+30:ih+10:color=white[t1]; \
 [2:v]pad=iw+30:ih+10:color=white[t2]; \
 [3:v]pad=iw+30:ih+10:color=white[t3]; \
 [4:v]pad=iw+30:ih+10:color=white[t4]; \
 [5:v]pad=iw:ih+10:color=white[t5]; \
 [t0][t1][t2][t3][t4][t5]hstack=6[t]; \
 [6:v]pad=iw+30:ih:color=white[b0]; \
 [7:v]pad=iw+30:ih:color=white[b1]; \
 [8:v]pad=iw+30:ih:color=white[b2]; \
 [9:v]pad=iw+30:ih:color=white[b3]; \
 [10:v]pad=iw+30:ih:color=white[b4]; \
 [b0][b1][b2][b3][b4][11:v]hstack=6[b]; \
 [t][b]vstack" \
MMNIST_FutureGAN_GroundTruth.gif

ffmpeg -i .//kth/sequence0001_FutureGAN.gif -i .//kth/sequence0002_FutureGAN.gif -i .//kth/sequence0003_FutureGAN.gif -i .//kth/sequence0004_FutureGAN.gif -i .//kth/sequence0001_GroundTruth.gif -i .//kth/sequence0002_GroundTruth.gif -i .//kth/sequence0003_GroundTruth.gif -i .//kth/sequence0004_GroundTruth.gif -filter_complex \
"[0:v]pad=iw+10:ih+10:color=white[t0]; \
 [1:v]pad=iw+10:ih+10:color=white[t1]; \
 [2:v]pad=iw+10:ih+10:color=white[t2]; \
 [3:v]pad=iw:ih+10:color=white[t3]; \
 [t0][t1][t2][t3]hstack=4[t]; \
 [4:v]pad=iw+10:ih:color=white[b0]; \
 [5:v]pad=iw+10:ih:color=white[b1]; \
 [6:v]pad=iw+10:ih:color=white[b2]; \
 [b0][b1][b2][7:v]hstack=4[b]; \
 [t][b]vstack" \
KTH_FutureGAN_GroundTruth.gif

ffmpeg -i .//cityscapes/sequence0001_FutureGAN.gif -i .//cityscapes/sequence0003_FutureGAN.gif -i .//cityscapes/sequence0004_FutureGAN.gif -i .//cityscapes/sequence0005_FutureGAN.gif -i .//cityscapes/sequence0001_GroundTruth.gif -i .//cityscapes/sequence0003_GroundTruth.gif -i .//cityscapes/sequence0004_GroundTruth.gif -i .//cityscapes/sequence0005_GroundTruth.gif -filter_complex \
"[0:v]pad=iw+10:ih+10:color=white[t0]; \
 [1:v]pad=iw+10:ih+10:color=white[t1]; \
 [2:v]pad=iw+10:ih+10:color=white[t2]; \
 [3:v]pad=iw:ih+10:color=white[t3]; \
 [t0][t1][t2][t3]hstack=4[t]; \
 [4:v]pad=iw+10:ih:color=white[b0]; \
 [5:v]pad=iw+10:ih:color=white[b1]; \
 [6:v]pad=iw+10:ih:color=white[b2]; \
 [b0][b1][b2][7:v]hstack=4[b]; \
 [t][b]vstack" \
Cityscapes_FutureGAN_GroundTruth.gif

ffmpeg -i .//MMNIST_FutureGAN_GroundTruth.gif -i .//KTH_FutureGAN_GroundTruth.gif -i .//Cityscapes_FutureGAN_GroundTruth.gif -filter_complex \
"[0:v]pad=iw:ih+20:color=white[0]; \
 [1:v]pad=iw:ih+20:color=white[1]; \
 [2:v]pad=iw:ih:color=white[2]; \
 [0][1][2]vstack=3" \
Predictions_FutureGAN_GroundTruth.gif

ffmpeg -i .//kth/long-term_prediction_120frames/sequence0001_FutureGAN.gif -i .//kth/long-term_prediction_120frames/sequence0002_FutureGAN.gif -i .//cityscapes/long-term_prediction_25frames/sequence0001_FutureGAN.gif -i .//cityscapes/long-term_prediction_25frames/sequence0004_FutureGAN.gif -i .//kth/long-term_prediction_120frames/sequence0001_GroundTruth.gif -i .//kth/long-term_prediction_120frames/sequence0002_GroundTruth.gif -i .//cityscapes/long-term_prediction_25frames/sequence0001_GroundTruth.gif -i .//cityscapes/long-term_prediction_25frames/sequence0004_GroundTruth.gif -filter_complex \
"[0:v]pad=iw+10:ih+10:color=white[t0]; \
 [1:v]pad=iw+10:ih+10:color=white[t1]; \
 [2:v]pad=iw+10:ih+10:color=white[t2]; \
 [3:v]pad=iw:ih+10:color=white[t3]; \
 [t0][t1][t2][t3]hstack=4[t]; \
 [4:v]pad=iw+10:ih:color=white[b0]; \
 [5:v]pad=iw+10:ih:color=white[b1]; \
 [6:v]pad=iw+10:ih:color=white[b2]; \
 [b0][b1][b2][7:v]hstack=4[b]; \
 [t][b]vstack" \
LongTerm_FutureGAN_GroundTruth.gif
