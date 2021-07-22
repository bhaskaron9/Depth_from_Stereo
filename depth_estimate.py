from util import get_depth_map
from disparity import find_disp_NCC,find_disp_DP
import matplotlib.pyplot as plt
import argparse
import os
import cv2
parser = argparse.ArgumentParser(description='Stereo based Depth Estimation')
parser.add_argument('--im1', default='tsukuba_l.png', type=str, help='First Image Path')
parser.add_argument('--im2', default='tsukuba_r.png', type=str, help='Second Image Path')
parser.add_argument('--baseline', default=100, type=float, help='Baseline')
parser.add_argument('--focal-length', default=96, type=float, help='Focal Length')
parser.add_argument('--winsize', default=3, type=int, help='Window Size')
parser.add_argument('--disparity-range', default=16, type=int, help='Disparity Range')
parser.add_argument('--method', default=0, type=int, help='Choose a method for depth estimation: 0: NCC based, 1: Dynamic Programming')
parser.add_argument('--occ-cost', default=0.01, type=float, help='Occlusion cost for Dynamic Programming solution')
args = parser.parse_args()
assert os.path.exists(args.im1)
assert os.path.exists(args.im2)
if __name__ == '__main__':
	imgL = cv2.imread(args.im1, 0)
	imgR = cv2.imread(args.im2, 0)
	if args.method == 0:
		print("Using NCC")
		img_dispL = find_disp_NCC(imgL, imgR, args.winsize, args.disparity_range)
		plt.figure(figsize=(10, 5));
		plt.subplot(1, 2, 1);
		plt.title('Disparity Image (Left), window={}'.format( args.winsize));
		plt.imshow(img_dispL);
		plt.axis("off");
		img_depthL = get_depth_map(img_dispL, args.baseline, args.focal_length)
		plt.subplot(1, 2, 2);
		plt.title('Depth Image (Left), window={}'.format( args.winsize));
		plt.imshow(img_depthL);
		plt.axis("off");
	elif args.method == 1:
		print("Using DP")
		img_dispL = find_disp_DP(imgL, imgR, args.occ_cost, args.winsize, args.disparity_range)
		plt.figure(figsize=(10, 5));
		plt.subplot(1, 2, 1);
		plt.title('Disparity Image (Left), window={}'.format( args.winsize));
		plt.imshow(img_dispL);
		plt.axis("off");
		img_depthL = get_depth_map(img_dispL, args.baseline, args.focal_length)
		plt.subplot(1, 2, 2);
		plt.title('Depth Image (Left), window={}'.format( args.winsize));
		plt.imshow(img_depthL);
		plt.axis("off");