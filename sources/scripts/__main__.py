from imagproc_pkg import *
from pandas import read_csv, DataFrame, concat
from os import listdir
from os.path import join


parser = ResumeParser()

resumes_df = DataFrame(columns=['Contents', 'Boundings', 'ParseFlag'])
# IF current is above... 
#resumes_df = read_csv("AllResumes_.csv", index_col=["Unnamed: 0"])

files = [f for f in listdir(RESUMES_PATH) if f.endswith('.png')]
files.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3].split('.')[0])))


#------------------ SECTION TO BE ITERATED
current = 1

for i, file_name in enumerate(files, start=1):
    if i <= current or  (not (i <= 89)) :
        continue
    
    if i % 10 == 0 :
        resumes_df.to_csv("AllResumes_.csv")
    
    img = cv2.imread(resume(file_name))
    
    try:
        lines, flag_ = parser.CroppeImage(img)
        
        text_list = []
        if flag_ == 'PSM11' :
            text_str = pytesseract.image_to_string(img, config=r' --psm 11 --oem 2', lang='eng+fra')
            text_list.append(text_str[:-2])
        
        elif flag_ == 'PSM7' :
            for idx in range(len(lines)):
                cropped_img = retrieve_cropped(img, lines, idx)
                text_str = pytesseract.image_to_string(cropped_img, config=r' --psm 7 --oem 2', lang='eng+fra')
            
                text_list.append(text_str[:-2])

    except (ValueError, IndexError, SystemError, Exception) as e:
        print(type(e).__name__ + f"[{i}] : " + str(e))
        continue        
    
    temp_resume_df = DataFrame({'Contents': [text_list], 'Boundings': [lines], 'ParseFlag': flag_})
    temp_resume_df.index = [file_name]
    resumes_df = concat([resumes_df, temp_resume_df])
    
    print(f"Image {i} processed : [{flag_}] ")


resumes_df.to_csv(join(DATA_PATH,"AllResumes_.csv"))