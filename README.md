# DeformNet

This project is project B in bachlor's degree: Learning to Deform for Efficient Image Compression.

It was supervised by Tamar Rott Shaham, directly related to her work "Deformation Aware Image Compression" https://tomer.net.technion.ac.il/files/2018/03/DeformationAwareCompression_.pdf

Our goal was to implement the iterative process, proposed in the paper, in a non-space-invariant network.

## Folder structure

In order to run the training script, the data and files should be arranged as follows:

+ DeformNet (dir)
	+ figs (dir – necessary for training)
	+ net_files (dir – necessary for training)
	+ logs (dir – necessary for training)
	+ jp2 (net input dir – necessary for training jp2 data)
		+ training_data (dir)
			+ <image name> (dir)
				+ R=<#> (dir, R is the compression rate)
					Input.png  (original image)
					y.png (deformed image)
					ux.mat (ux flow)
					uy.mat (uy flow)
		+ test_data (dir)
			+ <image name> (dir)
				+ R=<#> (dir)
					Input.png  (original image)
					y.png (deformed image)
					ux.mat (ux flow)
					uy.mat (uy flow)
	+ Affine (net input dir – necessary for training affine data)
		+ training_data (dir)
			+ <image_name> (dir)
				Input.png  (original image)
				y.png (deformed image)
				ux.mat (ux flow)
				uy.mat (uy flow)
		+ test_data (dir)
			+ <image_name> (dir)
				Input.png  (original image)
				y.png (deformed image)
				ux.mat (ux flow)
				uy.mat (uy flow)
	+ StaticTransform (net input dir – necessary for training static flow data)
		+ training_data (dir)
			+ <image_name> (dir)
				Input.png  (original image)
				y.png (deformed image)
				ux.mat (ux flow)
				uy.mat (uy flow)
		+ test_data (dir)
			+ <image_name> (dir)
				Input.png  (original image)
				y.png (deformed image)
				ux.mat (ux flow)
				uy.mat (uy flow)
	+ Forward_Images (dir)
		+ <dataset name, should correspond path in the script> (dir)
			+ <image name> (dir)
				Input.png  (original image)
				y.png (deformed image)
				ux.mat (ux flow)
				uy.mat (uy flow)
	+ Forward_results (dir)
		(the content is automatically created by the script)			
	DataLoader.py
	Main.py
	Model.py
	Forward_main.py
	Forward_Dataloader.py

