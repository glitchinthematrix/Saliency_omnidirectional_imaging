import os
import sys
import shutil
import subprocess
import cv2
import tempfile
import time

from argparse import ArgumentParser
import numpy as np


from scipy import io, ndimage

from scipy import misc
from scipy import ndimage
from matplotlib import pyplot as plt

PATH = os.path.dirname(os.path.realpath(__file__))
CUBE2SPHERE_PATH = '{}\\cube2sphere\\cube2sphere.py'.format(PATH)
SPHERE2CUBE_PATH = '{}\\sphere2cube\\sphere2cube.py'.format(PATH)
SAM_PATH = '{}\\saliency_attentive_model\\'.format(PATH)
BMS_PATH = '{}\\BMS\\EXE\\'.format(PATH)
Op_path= "C:\\Users\\user\\Desktop\\Bhishma\\Final_model\\BMS\\EXE\\output\\"


def compute_SAM_saliency(img, verbose=False, **kwargs):
    proportions = float(img.shape[1]) / img.shape[0]
    mode = 'default'
    if abs(proportions - 2.0) < 1e-5:
        # 2:1
        mode = 'simple'
    elif abs(proportions - 1.0) < 1e-5:
        # 1:1
        mode = 'cubeface'
    if verbose:
       print ('Selected mode: {}'.format(mode),file=sys.stderr)
    config_file =  '{}/config'.format(SAM_PATH)

    assert mode in {'simple', 'cubeface', 'default'}
    config_file += '_' + mode
    config_file += '.py'

    config_dst = '{}/config.py'.format(SAM_PATH)
    shutil.copy(config_file, config_dst)
    in_dir = tempfile.mkdtemp() + '\\'
    misc.imsave('{}/input.png'.format(in_dir), img)
    cmd = '''python saliency_attentive_model\\main.py test {input_path} ; cd..'''.format( input_path=in_dir)
    
    if verbose:
        print (cmd,file=sys.stderr)
    os.system(cmd)
    result = subprocess.check_output(cmd, shell=True)
    print(result)
    expected_out_fname = os.path.join(SAM_PATH, 'predictions','input.png' )
    #res = h5py.File(expected_out_fname, 'r')['saliency_map'][:]
    print(expected_out_fname)
    res = cv2.imread(expected_out_fname,0)
   # os.remove(expected_out_fname)
    shutil.copy('{}/config_default.py'.format(SAM_PATH), config_dst)
    shutil.rmtree(in_dir)
    return res



def compute_BMS_saliency(img, verbose=False, **kwargs):
    in_dir = tempfile.mkdtemp() + '/'
    misc.imsave('{}/input.png'.format(in_dir), img)
   # cmd = '''cd {bms_path}; BMS.exe {input_path} output/ 8 7 9 9 2 1 400 ; cd.. ;cd..'''.format(bms_path = BMS_PATH, input_path = in_dir)
    cmd = "{bms_path}BMS.exe {source} {output} 8 7 9 9 2 1 400".format(bms_path=BMS_PATH, source = in_dir, output = Op_path)

    if verbose:
        print (cmd,file=sys.stderr)
    os.system(cmd)

    expected_out_fname =  os.path.join(BMS_PATH, 'output','input.png' )
    res = cv2.imread(expected_out_fname,0)
    os.remove(expected_out_fname)
    return res




def compute_saliency_map_360(img_360_path,
                             verbose=False,
                             **kwargs):
    tmp_dir = tempfile.mkdtemp()

    
    #Calculating middle part using SAM using 8 iterations
    #Calculating 180 and 0
    print("Started computing middle part")
    img_360 = misc.imread(img_360_path)
    img_360_small=cv2.resize(img_360,None,fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)
    smap_0 = compute_SAM_saliency(img_360_small, verbose =verbose, **kwargs)

    img_back = np.zeros(img_360_small.shape, dtype = img_360_small.dtype)
    half_width = int(img_back.shape[1] / 2)
    img_back[:, :half_width, :] = img_360_small[:, -half_width:, :]
    img_back[:, half_width:, :] = img_360_small[:, :-half_width, :]

    smap_180_initial = compute_SAM_saliency(img_back,verbose=verbose, **kwargs)
    smap_180 = np.zeros(smap_180_initial.shape)
    smap_180[:, :half_width] = smap_180_initial[:, -half_width:]
    smap_180[:, half_width:] = smap_180_initial[:, :-half_width]


    #Calculating 90 
    img_90 =  np.zeros(img_360_small.shape, dtype = img_360_small.dtype)
    width_four = int(img_90.shape[1] / 4)
    img_90[:,:width_four, :] = img_360_small[:, -width_four:,:]
    img_90[:,width_four: ,:] = img_360_small[:, :-width_four, :]

    smap_90_initial = compute_SAM_saliency(img_90,verbose =verbose, **kwargs) 
    smap_90 = np.zeros(smap_90_initial.shape)
    smap_90[:, :-width_four] = smap_90_initial[:,width_four:]
    smap_90[:,-width_four:] = smap_90_initial[:,:width_four]

    #Calculating 270
    img_270 =  np.zeros(img_360_small.shape, dtype = img_360_small.dtype)
    width_four = int(img_270.shape[1] / 4)
    img_270[:,:-width_four, :] = img_360_small[:, width_four:,:]
    img_270[:,-width_four: ,:] = img_360_small[:, :width_four, :]

    smap_270_initial = compute_SAM_saliency(img_270,verbose =verbose, **kwargs) 
    smap_270 = np.zeros(smap_270_initial.shape)
    smap_270[:, :width_four] = smap_270_initial[:,-width_four:]
    smap_270[:,width_four:] = smap_270_initial[:,:-width_four]
    
    #Calculating 45

    img_45 =  np.zeros(img_360_small.shape, dtype = img_360_small.dtype)
    width_eight = int(img_45.shape[1] / 8)
    img_45[:,:width_eight, :] = img_360_small[:, -width_eight:,:]
    img_45[:,width_eight: ,:] = img_360_small[:, :-width_eight, :]

    smap_45_initial = compute_SAM_saliency(img_45,verbose =verbose, **kwargs) 
    smap_45 = np.zeros(smap_45_initial.shape)
    smap_45[:, :-width_eight] = smap_45_initial[:,width_eight:]
    smap_45[:,-width_eight:] = smap_45_initial[:,:width_eight]

    #Calculating 315
    img_315 =  np.zeros(img_360_small.shape, dtype = img_360_small.dtype)
    width_eight = int(img_315.shape[1] / 8)
    img_315[:,:-width_eight, :] = img_360_small[:, width_eight:,:]
    img_315[:,-width_eight: ,:] = img_360_small[:, :width_eight, :]

    smap_315_initial = compute_SAM_saliency(img_315,verbose =verbose, **kwargs) 
    smap_315 = np.zeros(smap_315_initial.shape)
    smap_315[:, :width_eight] = smap_315_initial[:,-width_eight:]
    smap_315[:,width_eight:] = smap_315_initial[:,:-width_eight]


    #Calculating 135
    img_135 =  np.zeros(img_360_small.shape, dtype = img_360_small.dtype)
    width_eight3 = int(3*img_135.shape[1] / 8)
    img_135[:,:width_eight3, :] = img_360_small[:, -width_eight3:,:]
    img_135[:,width_eight3: ,:] = img_360_small[:, :-width_eight3, :]

    smap_135_initial = compute_SAM_saliency(img_135,verbose =verbose, **kwargs) 
    smap_135 = np.zeros(smap_135_initial.shape)
    smap_135[:, :-width_eight3] = smap_135_initial[:,width_eight3:]
    smap_135[:,-width_eight3:] = smap_135_initial[:,:width_eight3]

    #Calculating 225
    img_225 =  np.zeros(img_360_small.shape, dtype = img_360_small.dtype)
    width_eight3 = int(3*img_225.shape[1] / 8)
    img_225[:,:-width_eight3, :] = img_360_small[:, width_eight3:,:]
    img_225[:,-width_eight3: ,:] = img_360_small[:, :width_eight3, :]

    smap_225_initial = compute_SAM_saliency(img_225,verbose =verbose, **kwargs) 
    smap_225 = np.zeros(smap_225_initial.shape)
    smap_225[:, :width_eight3] = smap_225_initial[:,-width_eight3:]
    smap_225[:,width_eight3:] = smap_225_initial[:,:-width_eight3]
    print("Middle part computed")



    #Creating top_bottom maps
    resolution = 960
       
    cubemap_folder = 'cubemap/'
    cubemap_folder = os.path.join(tmp_dir, cubemap_folder)
    if not os.path.exists(cubemap_folder):
        os.mkdir(cubemap_folder)
    cmd = list(['python', SPHERE2CUBE_PATH, img_360_path, 
           '-o',  cubemap_folder, '-f', 'png',
           '-t', '3',
           '-r', str(resolution),
           '-R', '0', '0', '0','-b','C:\\Program Files\\Blender Foundation\\Blender\\blender.exe'])
    if verbose:
        print >> sys.stderr, ' '.join(cmd)
    subprocess.call(cmd)
    face_bottom_id = 5
    face_bottom_index = [3, 1, 4, 2, 6, 5].index(face_bottom_id)
    face_top_id = 6
    face_top_index = [3, 1, 4, 2, 6, 5].index(face_top_id)
    faces_order = ['{}/face_{}_{}.png'.format(cubemap_folder,
                                              i,
                                              resolution) 
                   for i in [3, 1, 4, 2, 6, 5]]
    sal_map_names = ['{}/face_{}_{}_sal_map.png'.format(cubemap_folder,
                                                        i,
                                                        resolution) 
                     for i in [3, 1, 4, 2, 6, 5]]

    bottom_face = misc.imread(faces_order[face_bottom_index])
    bottom_sal_map = compute_SAM_saliency(bottom_face, 
                                        verbose=verbose, **kwargs)
    other_sal_map = np.zeros(bottom_sal_map.shape)
    top_face = misc.imread(faces_order[face_top_index])
    top_sal_map = compute_SAM_saliency(top_face, 
                                     verbose=verbose, **kwargs)
    sal_maps = [bottom_sal_map if i == face_bottom_index else (top_sal_map if i == face_top_index else other_sal_map)
                        for i in range(6)]

    for face_name, face_sal in zip(sal_map_names, sal_maps):
        cv2.imwrite(face_name, face_sal)
    out_fname_template = '{}/out####.png'.format(cubemap_folder)
    out_fname = '{}/out0001.png'.format(cubemap_folder)
    cmd = ['python', CUBE2SPHERE_PATH,
               '-o', out_fname_template, '-f', 'png',
               '-t', '3',
               '-r', str(img_360.shape[1]), str(img_360.shape[0]),
               '-R', '0', '0', '180','-b','C:\\Program Files\\Blender Foundation\\Blender\\blender.exe']
    cmd += sal_map_names
    if verbose:
        print >> sys.stderr, ' '.join(cmd)
    subprocess.call(cmd)

    just_top_bottom_face_360 = cv2.imread(out_fname,0)

    print("Top bottom face has been computed")
    #Calculating CMP map using BMS

    cmd = ['python', SPHERE2CUBE_PATH, img_360_path, 
           '-o',  cubemap_folder, '-f', 'png',
           '-t', '3',
           '-r', str(resolution),
           '-R', '0', '0', '0','-b','C:\\Program Files\\Blender Foundation\\Blender\\blender.exe']
    if verbose:
        print >> sys.stderr, ' '.join(cmd)
    subprocess.call(cmd)

    faces_order = ['{}/face_{}_{}.png'.format(cubemap_folder,
                                                  i,
                                                  resolution) 
                       for i in [3, 1, 4, 2, 6, 5]]

    sal_map_names = ['{}/face_{}_{}_sal_map.png'.format(cubemap_folder,
                                                  i,
                                                  resolution) 
                       for i in [3, 1, 4, 2, 6, 5]]
    faces = [misc.imread(face_name) for face_name in faces_order]
    sal_maps = [compute_BMS_saliency(face,
                                    verbose=verbose, **kwargs) 
                   for face in faces]

    for face_name, face_sal in zip(sal_map_names, sal_maps):
        cv2.imwrite(face_name, face_sal)

    out_fname_template = '{}/out####.png'.format(cubemap_folder)
    out_fname = '{}/out0001.png'.format(cubemap_folder)
    cmd = ['python', CUBE2SPHERE_PATH,
               '-o', out_fname_template, '-f', 'png',
               '-t', '3',
               '-r', str(img_360.shape[1]), str(img_360.shape[0]),
               '-R', '0', '0', '180','-b','C:\\Program Files\\Blender Foundation\\Blender\\blender.exe']
    cmd += sal_map_names
    if verbose:
        print >> sys.stderr, ' '.join(cmd)
    subprocess.call(cmd)
    
    CMP_BMS000 = cv2.imread(out_fname,0)


    cmd = ['python', SPHERE2CUBE_PATH, img_360_path, 
           '-o',  cubemap_folder, '-f', 'png',
           '-t', '3',
           '-r', str(resolution),
           '-R', '45', '0', '0','-b','C:\\Program Files\\Blender Foundation\\Blender\\blender.exe']
    if verbose:
        print >> sys.stderr, ' '.join(cmd)
    subprocess.call(cmd)

    faces_order = ['{}/face_{}_{}.png'.format(cubemap_folder,
                                                  i,
                                                  resolution) 
                       for i in [3, 1, 4, 2, 6, 5]]

    sal_map_names = ['{}/face_{}_{}_sal_map.png'.format(cubemap_folder,
                                                  i,
                                                  resolution) 
                       for i in [3, 1, 4, 2, 6, 5]]
    faces = [misc.imread(face_name) for face_name in faces_order]
    sal_maps = [compute_BMS_saliency(face,
                                    verbose=verbose, **kwargs) 
                   for face in faces]

    for face_name, face_sal in zip(sal_map_names, sal_maps):
        cv2.imwrite(face_name, face_sal)

    out_fname_template = '{}/out####.png'.format(cubemap_folder)
    out_fname = '{}/out0001.png'.format(cubemap_folder)
    cmd = ['python', CUBE2SPHERE_PATH,
               '-o', out_fname_template, '-f', 'png',
               '-t', '3',
               '-r', str(img_360.shape[1]), str(img_360.shape[0]),
               '-R', '45', '0', '180','-b','C:\\Program Files\\Blender Foundation\\Blender\\blender.exe']
    cmd += sal_map_names
    if verbose:
        print >> sys.stderr, ' '.join(cmd)
    subprocess.call(cmd)
    
    CMP_BMS4500 = cv2.imread(out_fname,0)


    cmd = ['python', SPHERE2CUBE_PATH, img_360_path, 
           '-o',  cubemap_folder, '-f', 'png',
           '-t', '3',
           '-r', str(resolution),
           '-R', '0', '45', '0','-b','C:\\Program Files\\Blender Foundation\\Blender\\blender.exe']
    if verbose:
        print >> sys.stderr, ' '.join(cmd)
    subprocess.call(cmd)

    faces_order = ['{}/face_{}_{}.png'.format(cubemap_folder,
                                                  i,
                                                  resolution) 
                       for i in [3, 1, 4, 2, 6, 5]]

    sal_map_names = ['{}/face_{}_{}_sal_map.png'.format(cubemap_folder,
                                                  i,
                                                  resolution) 
                       for i in [3, 1, 4, 2, 6, 5]]
    faces = [misc.imread(face_name) for face_name in faces_order]
    sal_maps = [compute_BMS_saliency(face,
                                    verbose=verbose, **kwargs) 
                   for face in faces]

    for face_name, face_sal in zip(sal_map_names, sal_maps):
        cv2.imwrite(face_name, face_sal)

    out_fname_template = '{}/out####.png'.format(cubemap_folder)
    out_fname = '{}/out0001.png'.format(cubemap_folder)
    cmd = ['python', CUBE2SPHERE_PATH,
               '-o', out_fname_template, '-f', 'png',
               '-t', '3',
               '-r', str(img_360.shape[1]), str(img_360.shape[0]),
               '-R', '0', '45', '180','-b','C:\\Program Files\\Blender Foundation\\Blender\\blender.exe']
    cmd += sal_map_names
    if verbose:
        print >> sys.stderr, ' '.join(cmd)
    subprocess.call(cmd)
    
    CMP_BMS0450 = cv2.imread(out_fname,0)


    cmd = ['python', SPHERE2CUBE_PATH, img_360_path, 
           '-o',  cubemap_folder, '-f', 'png',
           '-t', '3',
           '-r', str(resolution),
           '-R', '0', '0', '45','-b','C:\\Program Files\\Blender Foundation\\Blender\\blender.exe']
    if verbose:
        print >> sys.stderr, ' '.join(cmd)
    subprocess.call(cmd)

    faces_order = ['{}/face_{}_{}.png'.format(cubemap_folder,
                                                  i,
                                                  resolution) 
                       for i in [3, 1, 4, 2, 6, 5]]

    sal_map_names = ['{}/face_{}_{}_sal_map.png'.format(cubemap_folder,
                                                  i,
                                                  resolution) 
                       for i in [3, 1, 4, 2, 6, 5]]
    faces = [misc.imread(face_name) for face_name in faces_order]
    sal_maps = [compute_BMS_saliency(face,
                                    verbose=verbose, **kwargs) 
                   for face in faces]

    for face_name, face_sal in zip(sal_map_names, sal_maps):
        cv2.imwrite(face_name, face_sal)

    out_fname_template = '{}/out####.png'.format(cubemap_folder)
    out_fname = '{}/out0001.png'.format(cubemap_folder)
    cmd = ['python', CUBE2SPHERE_PATH,
               '-o', out_fname_template, '-f', 'png',
               '-t', '3',
               '-r', str(img_360.shape[1]), str(img_360.shape[0]),
               '-R', '0', '0', '225','-b','C:\\Program Files\\Blender Foundation\\Blender\\blender.exe']
    cmd += sal_map_names
    if verbose:
        print >> sys.stderr, ' '.join(cmd)
    subprocess.call(cmd)
    
    CMP_BMS0045 = cv2.imread(out_fname,0)


    cmd = ['python', SPHERE2CUBE_PATH, img_360_path, 
           '-o',  cubemap_folder, '-f', 'png',
           '-t', '3',
           '-r', str(resolution),
           '-R', '45', '45', '0','-b','C:\\Program Files\\Blender Foundation\\Blender\\blender.exe']
    if verbose:
        print >> sys.stderr, ' '.join(cmd)
    subprocess.call(cmd)

    faces_order = ['{}/face_{}_{}.png'.format(cubemap_folder,
                                                  i,
                                                  resolution) 
                       for i in [3, 1, 4, 2, 6, 5]]

    sal_map_names = ['{}/face_{}_{}_sal_map.png'.format(cubemap_folder,
                                                  i,
                                                  resolution) 
                       for i in [3, 1, 4, 2, 6, 5]]
    faces = [misc.imread(face_name) for face_name in faces_order]
    sal_maps = [compute_BMS_saliency(face,
                                    verbose=verbose, **kwargs) 
                   for face in faces]

    for face_name, face_sal in zip(sal_map_names, sal_maps):
        cv2.imwrite(face_name, face_sal)

    out_fname_template = '{}/out####.png'.format(cubemap_folder)
    out_fname = '{}/out0001.png'.format(cubemap_folder)
    cmd = ['python', CUBE2SPHERE_PATH,
               '-o', out_fname_template, '-f', 'png',
               '-t', '3',
               '-r', str(img_360.shape[1]), str(img_360.shape[0]),
               '-R', '45', '45', '180','-b','C:\\Program Files\\Blender Foundation\\Blender\\blender.exe']
    cmd += sal_map_names
    if verbose:
        print >> sys.stderr, ' '.join(cmd)
    subprocess.call(cmd)
    
    CMP_BMS45450 = cv2.imread(out_fname,0)

    print("CMP_BMS map has been computed")

    CMP_BMS = np.add(0.2*CMP_BMS000,0.2*CMP_BMS4500)
    CMP_BMS = np.add(CMP_BMS,0.2*CMP_BMS0450)
    CMP_BMS = np.add(CMP_BMS,0.2*CMP_BMS0045)
    CMP_BMS = np.add(CMP_BMS,0.2*CMP_BMS45450)
    #computing the final map 
    #Combining Middle maps
    p1 = np.amax(smap_0)
    p2 = np.amax(smap_45)
    img=np.add(0.125*smap_0,0.125*p1/p2*smap_45)
    p3=np.amax(smap_90)
    img=np.add(img,0.125*p1/p3*smap_90)
    p4=np.amax(smap_135)
    img=np.add(img,0.125*p1/p4*smap_135)
    p8=np.amax(smap_180)
    img=np.add(img,0.125*p1/p8*smap_180)
    p9=np.amax(smap_225)
    img=np.add(img,0.125*p1/p9*smap_225)
    p10=np.amax(smap_270)
    img=np.add(img,0.125*p1/p10*smap_270)
    p11=np.amax(smap_315)
    img=np.add(img,0.125*p1/p11*smap_315)

    frames = []
    frames.append(smap_0)
    frames.append(smap_45)
    frames.append(smap_90)
    frames.append(smap_135)
    frames.append(smap_180)
    frames.append(smap_225)
    frames.append(smap_270)
    frames.append(smap_315)

    img_max=np.maximum(smap_0,p1/p2*smap_45) 
    img_max=np.maximum(img,p1/p3*smap_90)
    img_max=np.maximum(img,p1/p4*smap_135)
    img_max=np.maximum(img,p1/p8*smap_180)
    img_max=np.maximum(img,p1/p9*smap_225)
    img_max=np.maximum(img,p1/p10*smap_270)
    img_max=np.maximum(img,p1/p11*smap_315)

    img_median = np.median(frames, axis=0).astype(dtype=np.uint8)
    h1,w1 = CMP_BMS.shape
    h2,w2 = img.shape
    img=cv2.resize(img,None,fx=w1/w2, fy=h1/h2, interpolation = cv2.INTER_CUBIC)
    img_max=cv2.resize(img_max,None,fx=w1/w2, fy=h1/h2, interpolation = cv2.INTER_CUBIC)
    img_median=cv2.resize(img_median,None,fx=w1/w2, fy=h1/h2, interpolation = cv2.INTER_CUBIC)

    p_edge = np.amax(just_top_bottom_face_360)
    p_avg = np.amax(img)
    p_max = np.amax(img_max)
    p_median = np.amax(img_median)

    img_median = np.maximum(img_median,p_median/p_edge*just_top_bottom_face_360)
    img_max = np.maximum(img_max, p_max/p_edge*just_top_bottom_face_360)
    img = np.maximum(img, p_avg/p_edge)



    VLIM=cv2.imread('VLIM.png',0)
    h1,w1 = img.shape
    h2,w2 =VLIM.shape

    VLIM=cv2.resize(VLIM,None,fx=w1/w2, fy=h1/h2, interpolation = cv2.INTER_CUBIC)
    img=np.add(0.7*img,0.3*VLIM)
    img_median=np.add(0.7*img_median,0.3*VLIM)
    img_max=np.add(0.7*img_max,0.3*VLIM)
   

    images = []

    images.append(CMP_BMS)
    images.append(img)
    images.append(img_max)
    images.append(img_median)

    return images

def parse_args():
    parser = ArgumentParser('Saliency model')
    
    parser.add_argument('input', 
                        help='Path to 360-equirectangular RGB input image')
    parser.add_argument('output', 
                        help='Path to 360-equirectangular output saliency image')

    parser.add_argument('--verbose', '-v', default=False,
                        action='store_true',
                        help='''Whether to output runtime info. Some details will be outputed either way, 
                        but you can squash them by adding "2>/dev/null 1>/dev/null", without quotes, to the end of your ./360_aware.py ... command.''')
    return parser.parse_args()

def main():
    args = parse_args()

    images = compute_saliency_map_360(img_360_path = args.input, verbose= args.verbose)
    po = np.amax(images[0])
    p1 = np.amax(images[1])
    p2 = np.amax(images[2])
    p3 = np.amax(images[3])

    CMP_avg = np.add(0.5*images[0]*p1/po,0.5*images[1])
    CMP_max = np.add(0.5*images[0]*p2/po,0.5*images[2])
    CMP_median = np.add(0.5*images[0]*p3/po,0.5*images[3])

    output_path = args.output

    plt.imsave(output_path +'Final_avg.png', CMP_avg, cmap='jet')
    plt.imsave(output_path +'Final_max.png', CMP_max, cmap='jet')
    plt.imsave(output_path +'Final_median.png',CMP_median, cmap='jet')

    cv2.imwrite(output_path +'SM_avg.png',CMP_avg)
    cv2.imwrite(output_path +'SM_max.png',CMP_max)
    cv2.imwrite(output_path +'SM_median.png',CMP_median)

if __name__ == "__main__": 
    main()

























    

    





    
    

    
    








