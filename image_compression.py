import cPickle
import numpy as np
from numpy import linalg as la
from PIL import Image
import sys
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)


def main(argv):
	# example calls:
	# python2 image_compression.py c lena-grayscale.tif lena-compressed.pickle
	# python2 image_compression.py d lena-compressed.pickle lena-reconstructed.tif
	# python2 image_compression.py s lena-grayscale.tif comparison/lena-reconstructed.tif
	usage = ("Correct usage: python2 "+argv[0].split("/")[-1]+" <c/d/s> <input> <output>\n"+
			"    c/d/s: flag to choose compression, decompression, or sweep\n"+
			"    input: file to compress/decompress\n"+
			"    output: where to store the processed file")

	if len(argv) != 4:
		print(usage)
		return

	if argv[1] == "c":
		im = Image.open(argv[2]).convert("L")
		A = np.matrix(im)
		mn = min(A.shape)
		mx = max(A.shape)
		(U, s, Vt) = np.linalg.svd(A)
		# print(len([v for v in s if v > 100]))
		S = np.zeros(A.shape, dtype=complex)
		S[:mn, :mn] = np.diag(s)
		cr = 0.2 # float between 0 and 1 (inclusive) that determines the level of compression (higher ratio means less compressed)
		f = open(argv[3], "wb")
		cPickle.dump([U[:, :int(cr*U.shape[1])], S[:int(cr*mn), :int(cr*mn)], Vt[:int(cr*Vt.shape[0]), :]], f)
		# cPickle.dump([U[:, :210], S[:210, :210], Vt[:210, :]], f)
		f.close()
		return

	elif argv[1] == "d":
		f = open(argv[2], "rb")
		loaded = cPickle.load(f)
		f.close()
		U = loaded[0]
		S = loaded[1]
		Vt = loaded[2]
		A = np.dot(U, np.dot(S, Vt)).H
		im = Image.new("L", A.shape)
		pixels = im.load()
		for i in range(A.shape[0]):
			for j in range(A.shape[1]):
				pixels[i, j] = A[i, j]
		im.save(argv[3])
		return

	elif argv[1] == "s": 
		components = argv[3].split(".")
		orig_im = Image.open(argv[2]).convert("L")
		A = np.matrix(orig_im)
		mn = min(A.shape)
		mx = max(A.shape)
		(U, s, Vt) = np.linalg.svd(A)
		S = np.zeros(A.shape, dtype=complex)
		S[:mn, :mn] = np.diag(s)
		for i in range(100):
			cr = (i+1)/100.0 # what % of the data is kept, / 100
			tmp_U = U[:, :int(cr*U.shape[1])]
			tmp_S = S[:int(cr*mn), :int(cr*mn)]
			tmp_Vt = Vt[:int(cr*Vt.shape[0]), :]
			tmp_A = np.dot(tmp_U, np.dot(tmp_S, tmp_Vt)).H
			new_im = Image.new("L", tmp_A.shape)
			pixels = new_im.load()
			for i in range(tmp_A.shape[0]):
				for j in range(tmp_A.shape[1]):
					pixels[i, j] = tmp_A[i, j]
			new_im.save(components[0]+"_{0:.2f}.".format(cr)+"".join(components[1:]))

	else:
		print(usage)
		return


if __name__ == "__main__":
	main(sys.argv)
