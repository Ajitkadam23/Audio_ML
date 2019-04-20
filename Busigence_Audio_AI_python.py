#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##############################################################################################################################
import os
import pickle
import numpy as np
from scipy.io.wavfile import read
import sklearn.mixture 
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
# gender="male"
val=0
def get_MFCC(sr,audio):
    features = mfcc.mfcc(audio,sr, 0.025, 0.01, 13,appendEnergy = False)
    features = preprocessing.scale(features)
    return features
#path to training data
source   = "/home/ajit/Desktop/youtube/male"
dest='/home/ajit/Desktop/youtube/'
files    = [os.path.join(source,f) for f in os.listdir(source) if
             f.endswith('.wav')]
features = np.asarray(())
for f in files:
    sr,audio = read(f)
    vector   = get_MFCC(sr,audio)
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
import sklearn
gmm = sklearn.mixture.GaussianMixture(n_components = 8, max_iter = 200, covariance_type='diag',n_init = 3)
gmm.fit(features)
picklefile = f.split("/")[-2].split(".wav")[0]+".gmm"
import csv
#Create csv filecsvfile="/Users/arushigupta148/Desktop/"+gender+".csv"with open(csvfile, "w") as output:    writer = csv.writer(output, lineterminator='\n')    writer.writerows(features)#Add new column to csv filedf = pd.read_csv(gender+".csv")df[val] = valdf.to_csv(gender+".csv")
# csvfile="/home/ajit/Desktop/"+gender+".csv"
# with open(csvfile, 'w') as output:   
#     writer = csv.writer(output, lineterminator='\n')   
#     writer.writerows(features)
# df = pd.read_csv(gender+".csv")
# df[val] = val
# df.to_csv(gender+".csv")
pickle.dump(gmm,open(dest + picklefile,'wb'))
print('modeling completed for gender:',picklefile)
 
import os
import pickle
import numpy as np
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
def get_MFCC(sr,audio):
    features = mfcc.mfcc(audio,sr, 0.025, 0.01, 13,appendEnergy = False)
    feat     = np.asarray(())
    for i in range(features.shape[0]):
        temp = features[i,:]
        if np.isnan(np.min(temp)):
            continue
        else:
            if feat.size == 0:
                feat = temp
            else:
                feat = np.vstack((feat, temp))
    features = feat;
    features = preprocessing.scale(features)
    return features
from pylab import*
#path to test data
# sourcepath = "/home/ajit/Desktop/test_data/AudioSet/male_clips"
sourcepath='/home/ajit/Desktop/Busgene_predictions'
#path to saved models
modelpath  = "/home/ajit/Desktop/youtube/"    
 
gmm_files = [os.path.join(modelpath,fname) for fname in
              os.listdir(modelpath) if fname.endswith('.gmm')]
models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
genders   = [fname.split("/")[-1].split(".gmm")[0] for fname
              in gmm_files]
files     = [os.path.join(sourcepath,f) for f in os.listdir(sourcepath)
              if f.endswith(".wav")] 

with open('/home/ajit/Desktop/Busgene_predictions.csv',mode='w')as output:
    outwriter=csv.writer(output,delimiter=',')
    for f in files:
    #     print(f.split("/")[-1])
        arr=[]
        sr, audio  = read(f)
        features   = get_MFCC(sr,audio)
        scores     = None
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm    = models[i]         #checking with each model one by one
            scores = np.array(gmm.score(features))

            log_likelihood[i] = scores.sum()
    #         print(log_likelihood)
        print(log_likelihood)    
        winner = np.argmax(log_likelihood)
        arr.append(log_likelihood)
        arr.append(gender[winner])
        outwriter.writerow(arr)
        print("\tdetected as - ", genders[winner],"\n\tscores:female ",log_likelihood[0],",male ", log_likelihood[1],"\n")


# In[7]:


#For FFT Calculation of spitted audio files
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
rate, data = wav.read('/home/ajit/Desktop/Busgene_predictions/new13.wav')
fft_out = fft(data)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data, np.abs(fft_out))
plt.show()
#for spliting original file into small files to avoid memory error
from pydub import AudioSegment
t1=1538
t2=1543
t1 = t1 * 1000 #Works in milliseconds
t2 = t2 * 1000 
newAudio = AudioSegment.from_wav("/home/ajit/Desktop/test.wav")
newAudio = newAudio[t1:t2]
newAudio.export('/home/ajit/Desktop/Busgene_predictions/new13.wav', format="wav")


# In[8]:


# %pylab inline
# import os
# import pandas as pd
# import librosa
# import glob
# import librosa.display
# plt.figure(figsize=(12,4))
# data,sampling_rate=librosa.load('/home/ajit/Desktop/Audio_1/sample_debate.mp3')
# librosa.display.waveplot(data,sr=sampling_rate)


# In[8]:


# import numpy as np


# In[7]:


# import argparse
# import sys
# import re
# import os.path as op
# import numpy as np
# from scipy.linalg import det
# from scipy.linalg import pinv


# def parse_recipe(rfile):
#     """Parses input recipe, checks for LNA's"""
#     r = []
#     audio_file = re.compile('audio=(\S+)')
#     lna_name = re.compile('lna=(\S+)')
#     start_time = re.compile('start-time=(\d+.\d+)')
#     end_time = re.compile('end-time=(\d+.\d+)')
#     for line in rfile:
#         try:
#             audio = audio_file.search(line).groups()[0]
#             lna = lna_name.search(line).groups()[0]
#             start = float(start_time.search(line).groups()[0])
#             end = float(end_time.search(line).groups()[0])
#             r.append((audio, lna, start, end))
#         except AttributeError:
#             print('Recipe line without recognizable data:')
#             print(line)
#     return r


# def load_features(recipeLine, fpath, ext):
#     """Load features from file"""
#     ffile_name = '/home/ajit/Desktop/Audio_1/sample_debate.mp3'
#     ffile_name += op.splitext(op.basename(recipeLine[0]))[0]
#     ffile_name += ext
#     #print 'Loading features from:', ffile_name
#     with open(ffile_name, 'rb') as ffile:
#         dim = int(np.fromfile(ffile, dtype=np.int32, count=1))
#         features = np.fromfile(ffile, dtype=np.float32)
#     #print 'Total features read:', features.size
#     features = features.reshape((features.size / dim), dim)
#     #print 'Final shape:', features.shape
#     return dim, features


# def get_spk_features(spk, features):
#     arr = features[int(spk[0][0]):int(spk[0][1])]
#     for s in spk[1:]:
#         # TODO: This copies, should be much faster and less memory consuming
#         # with views of the features, same everywhere else
#         arr = np.concatenate((arr, features[int(s[0]):int(s[1])]))
#     return arr


# def write_recipe_line(recline, start, end, lna_start, speaker, outf, segf=None):
#     """Write output recipes"""
#     global lna_letter, lna_count
#     lna = recline[1]
#     if not args.dlr:
#         if lna[:lna.find('_')] == lna_letter:
#             lna_count += 1
#         else:
#             lna_count = 1
#             lna_letter = lna[:lna.find('_')]
#         lna = lna[:lna.find('_') + 1] + str(lna_count)
#     outf.write('audio=' + recline[0] +
#                ' lna=' + lna +
#                ' start-time=' + str(start / rate + lna_start) +
#                ' end-time=' + str(end / rate + lna_start) +
#                ' speaker=speaker_' + str(speaker) + '\n')
#     if segpath != '' and segf is not None:
#         alignment = ' alignment=' + segpath + lna + '.seg'
#         segf.write('audio=' + recline[0] +
#                    alignment +
#                    ' lna=' + lna +
#                    ' start-time=' + str(start / rate + lna_start) +
#                    ' end-time=' + str(end / rate + lna_start) +
#                    ' speaker=speaker_' + str(speaker) + '\n')


# def bic(arr1, arr2):
#     """Bayes Information Criterion."""
#     # Notes: In the seminal paper "Speakers, environment and channel
#     # change detection and clustering via the Bayesian Information
#     # Criterion" by Chen and Gopalakrishnan, they use a growing window
#     # approach, so it's not directly comparable when using a fixed
#     # sliding window.
#     arr = np.concatenate((arr1, arr2))
#     N1 = arr1.shape[0]
#     N2 = arr2.shape[0]
#     S1 = np.cov(arr1, rowvar=0)
#     S2 = np.cov(arr2, rowvar=0)
#     N = arr.shape[0]
#     S = np.cov(arr, rowvar=0)
#     d = 0.5 * N * np.log(det(S)) - 0.5 * N1 * np.log(det(S1))\
#         - 0.5 * N2 * np.log(det(S2))
#     p = arr.shape[1]
#     corr = args.lambdac * 0.5 * (p + 0.5 * p * (p + 1)) * np.log(N)
#     d -= corr
#     return d


# def glr(arr1, arr2):
#     """Generalized Likelihood Ratio"""
#     N1 = arr1.shape[0]
#     N2 = arr2.shape[0]
#     S1 = np.cov(arr1, rowvar=0)
#     S2 = np.cov(arr2, rowvar=0)
#     N = float(N1 + N2)
#     # This is COV only version, not optimized (revise) but more robust
#     # to environment noise conditions.
#     # See Ulpu thesis pages 30-31, also Gish et al. "Segregation of
#     # Speakers for Speech Recognition and Speaker Identification"
#     d = -(N / 2.0) * ((N1 / N) * np.log(det(S1)) + (N2 / N) * np.log(det(S2))
#                       - np.log(det((N1 / N) * S1 + (N2 / N) * S2)))
#     # Ulpu version:
#     # Includes the mean, theoretically less robust
#     # arr = features[start:start+2*winsize]
#     # S = cov(arr, rowvar=0)
#     # d = -0.5*(N1*log(det(S1))+N2*log(det(S2))-N*log(det(S)))
#     return d


# def kl2(arr1, arr2):
#     """Simmetric Kullback-Leibler distance"""
#     S1 = np.cov(arr1, rowvar=0)
#     S2 = np.cov(arr2, rowvar=0)
#     m1 = np.mean(arr1, 0)
#     m2 = np.mean(arr2, 0)
#     delta = m1 - m2
#     d = 0.5 * np.trace((S1 - S2) * (pinv(S2) - pinv(S1))) +\
#         0.5 * np.trace((pinv(S1) + pinv(S2)) * delta * delta.T)
#     return d


# def spk_cluster_in(features, recline, speakers, outf, dist=bic, segf=None):
#     """Clusters same speaker turns"""
#     global total_segments
#     global max_dist, min_dist, max_det_dist, min_det_dist
#     start = int(recline[2] * rate)
#     end = int(recline[3] * rate)
#     arr2 = features[start:end]
#     mind = sys.maxint
#     spk = 0
#     while spk < len(speakers):
#         arr1 = get_spk_features(speakers[spk], features)
#         # print start, end, speakers[spk], arr1.shape, arr2.shape
#         d = dist(arr1, arr2)
#         if args.tt:
#             print('Time:', end, '- Distance:', d, '- Speaker:', spk + 1)
#         # Ignore infinite distances (non-speech?) and record stats
#         if d != np.inf and d != -np.inf:
#             if d > max_dist:
#                 max_dist = d
#             if d < min_dist:
#                 min_dist = d
#             if d < mind:
#                 mind = d
#                 best_candidate = spk
#         spk += 1
#     if mind <= threshold:
#         # Negative, same speaker!!
#         # Stats should be of total detected speakers
#         if d > max_det_dist:
#             max_det_dist = d
#         if d < min_det_dist:
#             min_det_dist = d
#         speakers[best_candidate].append((start, end))
#         # best_candidate + 1 because we want speakers_ >= 1 in the output
#         write_recipe_line(recline, start, end, 0, best_candidate + 1,
#                           outf, segf)
#     else:  # Positive, new speaker!
#         speakers.append([(start, end)])
#         write_recipe_line(recline, start, end, 0, len(speakers),
#                           outf, segf)


# def spk_cluster_hi(features, recipe, speakers, outf, dist=bic, segf=None):
#     """Clusters same speaker turns"""
#     global total_segments
#     global max_dist, min_dist, max_det_dist, min_det_dist
#     sp = len(speakers)
#     # distances = np.empty((sp, sp), dtype=int)
#     distances = np.empty((sp, sp))
#     np.fill_diagonal(distances, sys.maxint)
#     mind = sys.maxint
#     # Get all initial distances
#     for s1 in xrange(sp):
#         for s2 in xrange(s1 + 1, sp):
#             arr1 = get_spk_features(speakers[s1], features)
#             arr2 = get_spk_features(speakers[s2], features)
#             d = dist(arr1, arr2)
#             distances[s1][s2] = d
#             distances[s2][s1] = d
#             # Ignore infinite distances (non-speech?) and record stats
#             if d != np.inf and d != -np.inf:
#                 if d > max_dist:
#                     max_dist = d
#                 if d < min_dist:
#                     min_dist = d
#     while True:
#         # Get min d
#         mind = distances.min()
#         index = distances.argmin()
#         best_candidates = (index / len(speakers), index % len(speakers))
#         best_candidates = (min(best_candidates), max(best_candidates))
#         if mind <= threshold or (args.max_spk > 0
#                                  and len(speakers) > args.max_spk):
#             # Negative, fuse speakers!!
#             if mind > max_det_dist:
#                 max_det_dist = mind
#             if mind < min_det_dist:
#                 min_det_dist = mind
#             print('Merging:', best_candidates[0] + 1, 'and',\
#                   best_candidates[1] + 1, 'distance:', mind)
#             speakers[best_candidates[0]].extend(speakers[best_candidates[1]])
#             speakers.pop(best_candidates[1])
#             # Recalculating new speaker distances vs rest
#             s1 = best_candidates[0]
#             s1b = best_candidates[1]
#             # s1b is "out"
#             distances = np.delete(distances, s1b, 0)
#             distances = np.delete(distances, s1b, 1)
#             for s2 in xrange(len(speakers)):
#                 if s2 == s1:
#                     continue
#                 arr1 = get_spk_features(speakers[s1], features)
#                 arr2 = get_spk_features(speakers[s2], features)
#                 d = dist(arr1, arr2)
#                 distances[s1][s2] = d
#                 distances[s2][s1] = d
#                 # Ignore infinite distances (non-speech?) and record stats
#                 if d != np.inf and d != -np.inf:
#                     if d > max_dist:
#                         max_dist = d
#                     if d < min_dist:
#                         min_dist = d
#         else:
#             # Convergence
#             break
#     # All done, time to write the output recipe
#     print('Final speakers:', len(speakers))
#     while True:
#         # TODO: Sloooowww
#         candidate = None
#         for s in xrange(len(speakers)):
#             if candidate is None and speakers[s] != []:
#                 candidate = (s, min(speakers[s]))
#             elif speakers[s] != []:
#                 candidate2 = min(speakers[s])
#                 if candidate2 < candidate[1]:
#                     candidate = (s, candidate2)
#         if candidate is None:
#             # No more to write
#             break
#         else:
#             speakers[candidate[0]].remove(candidate[1])
#             write_recipe_line(recipe[candidate[1][2]], candidate[1][0],
#                               candidate[1][1], 0, candidate[0] + 1, outf,
#                               segf)


# def process_recipe(recipe, speakers, outf, segf=None):
#     """Process recipe, outputs a new recipe"""
#     this_wav = ''
#     l = 0
#     while l < len(recipe):
#         if recipe[l][0] != this_wav:
#             this_wav = recipe[l][0]
#             feas = load_features(recipe[l], feapath, feaext)
#             # Should I empty detected speakers here for a new wav?  Maybe not,
#             # if batch processing in the same recipe the wavs should be
#             # related
#         if speakers == [] and args.method == 'in':
#             speakers.append([(recipe[l][2] * rate, recipe[l][3] * rate)])
#             write_recipe_line(recipe[l], recipe[l][2] * rate,
#                               recipe[l][3] * rate, 0, len(speakers),
#                               outf, segf)
#         elif args.method == 'hi':
#             # Populate for initial clustering
#             speakers.append([(recipe[l][2] * rate, recipe[l][3] * rate, l)])
#         else:
#             # args.method == 'in', after first speaker initialization
#             spk_cluster_m(feas[1], recipe[l], speakers, outf,
#                           dist, segf)
#         l += 1
#     if args.method == 'hi':
#         # Initial clustering done, ready to start
#         # TODO: Multiple wavs on same recipe fails
#         print('Initial cluster with:', len(speakers), 'speakers')
#         spk_cluster_m(feas[1], recipe, speakers, outf,
#                       dist, segf)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Perform speaker clustering,\
#              using a distance measure.')
#     parser.add_argument('recfile', type=str,
#                         help='Specifies the input recipe file')
#     parser.add_argument('feapath', type=str,
#                         help='Specifies the features files path')
#     parser.add_argument('-seg', dest='segpath', type=str, default='',
#                         help='Specifies the alignment segmentation files path\
#                         and generates "alignment=" information, default empty\
#                         (not generate)')
#     parser.add_argument('-o', dest='outfile', type=str, default=sys.stdout,
#                         help='Specifies an output file, default stdout. If\
#                         specified with the "-seg" option, a second output file\
#                         will be created with "-seg" appended to the name\
#                         before the extension')
#     parser.add_argument('-fe', dest='feaext', type=str, default='.fea',
#                         help='Specifies feature file extension, default ".fea"')
#     parser.add_argument('-se', dest='segext', type=str, default='.seg',
#                         help='Specifies segmentation files extension, default ".seg"')
#     parser.add_argument('-f', dest='frame_rate', type=int, default=125,
#                         help='Specifies the frame rate, default 125')
#     parser.add_argument('-m', dest='method', type=str,
#                         choices=['in', 'hi'], default='hi',
#                         help='Specifies the clustering method, hierarchical\
#                         agglomerative or in-order consecutive clustering.\
#                         Default hierarchical (slower but more accurate).')
#     parser.add_argument('-d', dest='distance', type=str,
#                         choices=['GLR', 'BIC', 'KL2'], default='BIC',
#                         help='Sets the distance measure to use, defaults to\
#                         Bayesian Information Criterion (BIC). Generalized\
#                         Likelihood Ration (GLR) or symmetric Kullback-Leibler\
#                         (KL2) are other possibilities.')
#     parser.add_argument('-t', dest='threshold', type=float, default=0.0,
#                         help='Specifies threshold distance for detection,\
#                         default 0.0 (nonsensical handpicked, tune it except\
#                         for BIC).')
#     parser.add_argument('-ms', dest='max_spk', type=int, default=0,
#                         help='Specifies the maximum speakers stopping criteria\
#                         for hierarchical clustering, default 0 (use only the\
#                         threshold as stopping criteria')
#     parser.add_argument('-l', dest='lambdac', type=float, default=1.3,
#                         help='Lambda penalty weight for BIC, default 1.3')
#     parser.add_argument('-tt', action='store_true',
#                         help='If set, outputs all the decision thresholds in\
#                         every clustering attempt, useful to define a proper\
#                         threshold.')
#     parser.add_argument('-dlr', action='store_true',
#                         help='If set, disables lna renaming, so it keeps the lna\
#                         original names (if there are two speakers in the same\
#                         lna, start and end line should be used for adaptation).\
#                         By default it renames so that all segments have a\
#                         a different rna name.')
#     args = parser.parse_args()

#     # Process arguments
#     print('Reading recipe from:', args.recfile)
#     with open(args.recfile, 'r') as recfile:
#         parsed_recipe = parse_recipe(recfile)

#     print('Reading feature files from:', args.feapath)
#     feapath = args.feapath
#     if feapath[-1] != '/':
#         feapath += '/'

#     if args.segpath != '':
#         print('Setting alignment segmentation files path to:', args.segpath)
#         if args.segpath[-1] != '/':
#             args.segpath += '/'
#         print('Segmentation files extension:', args.segext)
#     segpath = args.segpath
#     segext = args.segext

#     print('Feature files extension:', args.feaext)
#     feaext = args.feaext

#     if args.outfile != sys.stdout:
#         outfile = args.outfile
#         print('Writing output to:', args.outfile)
#         if segpath != '':
#             segfile = op.splitext(outfile)[0]
#             segfile += '-seg' + op.splitext(outfile)[1]
#             print('Writing seg output to:', segfile)
#         else:
#             segfile = False
#     else:
#         outfile = sys.stdout
#         print('Writing output to: stdout')

#     rate = float(args.frame_rate)
#     print('Conversion rate set to frame rate:', rate)

#     if args.method == 'hi':
# #         print 'Using hierarchical clustering'
#         spk_cluster_m = spk_cluster_hi
#     elif args.method == 'in':
# #         print 'Using in-order consecutive clustering'
#         spk_cluster_m = spk_cluster_in

#     if args.distance == 'GLR':
# #         print 'Using GLR as distance measure'
#         dist = glr
#     elif args.distance == 'BIC':
# #         print 'Using BIC as distance measure, lambda =', args.lambdac
#         dist = bic
#     elif args.distance == 'KL2':
# #         print 'Using KL2 as distance measure'
#         dist = kl2

# #     print 'Threshold distance:', args.threshold
#     threshold = args.threshold
# #     print 'Maximum speakers:', args.max_spk

#     lna_letter = 'a'
#     lna_count = 0
#     if args.dlr:
#         print('Disabling LNA renaming')

#     # End of argument processing

#     # Some useful metrics
#     max_dist = 0
#     min_dist = sys.maxint
#     max_det_dist = 0
#     min_det_dist = sys.maxint

#     # Detected speakers
#     speakers = []

#     # Do the real work
#     if outfile != sys.stdout:
#         with open(outfile, 'w') as outf:
#             if segfile:
#                 with open(segfile, 'w') as segf:
#                     process_recipe(parsed_recipe, speakers, outf, segf)
#             else:
#                 process_recipe(parsed_recipe, speakers, outf)

#     else:
#         process_recipe(parsed_recipe, speakers, outfile)

#     print('Useful metrics for determining the right threshold:')
#     print('---------------------------------------------------')
#     print('Maximum between segments distance:', max_dist)
#     if min_dist < sys.maxint:
#         print('Minimum between segments distance:', min_dist)
#     print('Total segments:', len(parsed_recipe))
# # print('Total detected speakers:', len(speakers))


# In[1]:


# x, fs, nbits = audiolab.wavread('/tmp/mozilla_ajit0/sample_debate.wav')
# audiolab.play(x, fs)
# N = 4*fs    # four seconds of audio
# X = scipy.fft(x[:N])
# Xdb = 20*scipy.log10(scipy.absolute(X))
# f = scipy.linspace(0, fs, N, endpoint=False)
# pylab.plot(f, Xdb)
# pylab.xlim(0, 5000)   # view up to 5 kHz
# # 
# Y = X*H
# y = scipy.real(scipy.ifft(Y))


# In[7]:


# import numpy as np
# import soundfile as sf
# import argparse
# import os
# import keras
# import sklearn
# import librosa

# eps = np.finfo(np.float).eps

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='Load keras model and predict speaker count'
#     )
#     parser.add_argument(
#         'audio',
#         help='audio file (16 kHz) of 5 seconds duration'
#     )

#     args = parser.parse_args()

#     # load model
#     model = keras.models.load_model(
#         os.path.join('models', 'RNN_keras2.h5')
#     )

#     # print model configuration
#     model.summary()

#     # load standardisation parameters
#     scaler = sklearn.preprocessing.StandardScaler()
#     with np.load(os.path.join("models", 'scaler.npz')) as data:
#         scaler.mean_ = data['arr_0']
#         scaler.scale_ = data['arr_1']

#     # compute audio
#     audio, rate = sf.read(args.audio, always_2d=True)

#     # downmix to mono
#     audio = np.mean(audio, axis=1)

#     # compute STFT
#     X = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T

#     # apply standardization
#     X = scaler.transform(X)

#     # cut to input shape length (500 frames x 201 STFT bins)
#     X = X[:model.input_shape[1], :]

#     # apply normalization
#     Theta = np.linalg.norm(X, axis=1) + eps
#     X /= np.mean(Theta)

#     # add sample dimension
#     Xs = X[np.newaxis, ...]

#     # predict output
#     ys = model.predict(Xs, verbose=0)
# print("Speaker Count Estimate: ", np.argmax(ys, axis=1)[0])


# In[8]:


# from os import path
# from pydub import AudioSegment

# # files                                                                         
# src = "/home/ajit/Desktop/Audio_1/sample_debate.mp3"
# dst = "/home/ajit/Desktop/test.wav"

# # convert wav to mp3                                                            
# sound = AudioSegment.from_mp3(src)
# # sound.export(dst, format="wav")


# In[3]:


# import sys

# from numpy import *
# from scipy import signal
# import scipy.io.wavfile

# from matplotlib import pyplot

# import sklearn.decomposition

# def main():
# 	# First load the audio data, the audio data on this example is obtained from http://www.ism.ac.jp/~shiro/research/blindsep.html
# 	rate, source = scipy.io.wavfile.read('/home/ajit/Desktop/newSong.wav')

# 	# The 2 sources are stored in left and right channels of the audio
# 	source_1, source_2 = source[:, 0], source[:, 1]
# 	data = c_[source_1, source_2]

# 	# Normalize the audio from int16 range to [-1, 1]
# 	data = data / 2.0 ** 15

# 	# Perform Fast ICA on the data to obtained separated sources
# 	fast_ica  = sklearn.decomposition.FastICA( n_components=2  )
# 	separated = fast_ica.fit_transform( data )

# 	# Check, data = separated X mixing_matrix + mean
# 	assert allclose( data, separated.dot( fast_ica.mixing_.T ) + fast_ica.mean_ )

# 	# Map the separated result into [-1, 1] range
# 	max_source, min_source = 1.0, -1.0
# 	max_result, min_result = max(separated.flatten()), min(separated.flatten())
# 	separated = map( lambda x: (2.0 * (x - min_result))/(max_result - min_result) + -1.0, separated.flatten() )
# 	separated = reshape( separated, (shape(separated)[0] / 2, 2) )
	
# 	# Store the separated audio, listen to them later
# 	scipy.io.wavfile.write( '/home/ajit/Desktop/newSong1.wav', rate, separated[:, 0] )
# 	scipy.io.wavfile.write( '/home/ajit/Desktop/newSong2.wav', rate, separated[:, 1] )

# 	# Plot the original and separated audio data
# 	fig = pyplot.figure( figsize=(10, 8) )
# 	fig.canvas.set_window_title( 'Blind Source Separation' )

# 	ax = fig.add_subplot(221)
# 	ax.set_title('Source #1')
# 	ax.set_ylim([-1, 1])
# 	ax.get_xaxis().set_visible( False )
# 	pyplot.plot( data[:, 0], color='r' )

# 	ax = fig.add_subplot(223)
# 	ax.set_ylim([-1, 1])
# 	ax.set_title('Source #2')
# 	ax.get_xaxis().set_visible( False )
# 	pyplot.plot( data[:, 1], color='r' )

# 	ax = fig.add_subplot(222)
# 	ax.set_ylim([-1, 1])
# 	ax.set_title('Separated #1')
# 	ax.get_xaxis().set_visible( False )
# 	pyplot.plot( separated[:, 0], color='g' )

# 	ax = fig.add_subplot(224)
# 	ax.set_ylim([-1, 1])
# 	ax.set_title('Separated #2')
# 	ax.get_xaxis().set_visible( False )
# 	pyplot.plot( separated[:, 1], color='g' )
# 	pyplot.show()



# if __name__ == '__main__':
#     main()


# In[53]:





# In[54]:





# In[55]:





# In[56]:





# In[57]:





# In[ ]:





# In[62]:





# In[32]:





# In[20]:





# In[ ]:




