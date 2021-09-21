import csv

with open('hiro_calibration_camera_xls.csv', 'r') as infile:
    reader = csv.reader(infile)
    with open('cam_locs.csv', 'a+') as outfile:
        writer = csv.writer(outfile)
        next(reader) # clear header
        for row in reader:
            writer.writerow((row[0],row[1],row[3],row[4]))