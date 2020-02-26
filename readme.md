

					ZoneMapAltCnt Implementation Readme
					
				
		This tool consists an implementation of three series of paper based on the ZoneMap Metric Calculation. The three papers are as below:
		
				1) ZoneMap(https://bit.ly/2QSE3on)
				2) ZoneMapAlt(https://bit.ly/389ruuF)
				3) ZoneMapAltCnt(https://bit.ly/2NqvSxo)
		
		The purpose of this tool is to evaluate the performance of an OCR engine in terms of detection as well as recognition performance.
		The tool follows the folder structure as below:
		
			root
				ground-truth
				images
				ocr-detection
				run.py
			
		The 'images' folder consists of document images whereas the 'ground-truth' and 'ocr-detection' folder consist of the ground truth and ocr detection
		results annotation respectively. These annotations need to be in xml format followed by labelImg tool. Each object in the annotations represents
		a word i.e all annotations are on a word level. The names of the images and the annotations files in both the folders must be the same. For
		example, for an image file in images folder with the name 'ABC.png' there has to be 'ABC.xml' in ground-truth and ocr-detection folders. These two
		ABC.xml files will differ in the bounding box and text values for each word object.
		
		
		Required Packages:
		
			1) Pandas
			2) lxml
			3) Shapely
			
		Usage:
		
		Place all the required files as described above in their respective folders. Execute the run.py file.
		After completion, the script will produce a 'results.csv' file in the root folder.
		
		The results.csv file will have the following information:
		
			1) file_id - name of image file sans extension.
			2) zonemap - Overall detection score. Values higher than 100 indicate large false alarms.
			3) match, merge, split, miss, false_alarm, multiple - Contribution of each detection category to the overall score.
			4) precision, recall - Overall text recognition score. Represents the character level accuracy.
			5) n_match, n_merge, n_split, n_miss, n_false_alarm, n_multiple - Number of cases in each category.
			
		
		
		#TODO: 1) Implementation of word detection boxes visualization.
			   2) Logger for debugging of file wise word cases.
