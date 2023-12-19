import os
import random
import shutil
import numpy as np

#from utils.utils import get_classes

#--------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
#   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
#   annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt
#   annotation_mode为3代表生成VOCdevkit文件夹及其文件
#--------------------------------------------------------------------------------------------------------------------------------#
annotation_mode     = 0
#-------------------------------------------------------------------#
#   必须要修改，用于生成2007_train.txt、2007_val.txt的目标信息
#   与训练和预测所用的classes_path一致即可
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#   仅在annotation_mode为0和2的时候有效
#-------------------------------------------------------------------#
classes_path        = 'model_data/voc_classes.txt'
#--------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
#   仅在annotation_mode为0和1的时候有效
#--------------------------------------------------------------------------------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.9
#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
VOCdevkit_path  = 'VOCdevkit'

VOCdevkit_sets  = [('2007', 'train'), ('2007', 'val')]

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

classes, _      = get_classes(classes_path)

#-------------------------------------------------------#
#   统计目标数量
#-------------------------------------------------------#
photo_nums  = np.zeros(len(VOCdevkit_sets))
nums        = np.zeros(len(classes))




def convert_annotation(year, image_id, list_file):
    #in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.txt'%(year, image_id)), encoding='utf-8')
    in_file = os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.txt'%(year, image_id))

    with open(in_file, 'r') as file:
            for line in file:
                words = line.split()
                cls = words[0]
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)
                if cls in classes:
                    try:
                        numbers = [int(word) for word in words[1:5]]
                        b = [numbers[0],numbers[1],numbers[0]+numbers[2],numbers[1]+numbers[3]]
                        #numbers_list.extend(numbers)
                    except ValueError:
                        pass
                list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
                nums[classes.index(cls)] = nums[classes.index(cls)] + 1
 
def extract_images(source_folder, target_folder):
    # 创建目标文件夹
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 支持的图像文件扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    # 遍历源文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            # 检查文件是否为图像文件
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                source_path = os.path.join(root, filename)
                #target_path = os.path.join(target_folder, os.path.relpath(source_path, source_folder))
                target_path = os.path.join(target_folder, filename)

                # 复制图像文件到目标文件夹
                shutil.copy2(source_path, target_path)

def extract_Annno(source_folder, target_folder):
    if os.path.exists(target_folder):
        os.makedirs(target_folder)

    Annno_extensions = ['.txt']

    # 遍历源文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            # 检查文件是否为图像文件
            if any(filename.lower().endswith(ext) for ext in Annno_extensions):
                source_path = os.path.join(root, filename)
                #target_path = os.path.join(target_folder, os.path.relpath(source_path, source_folder))
                target_path = os.path.join(target_folder, filename)

                # 创建目标文件夹，确保目录存在
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                # 复制图像文件到目标文件夹
                shutil.copy2(source_path, target_path)

if __name__ == "__main__":
    random.seed(0)

    if annotation_mode == 0 or annotation_mode == 3:
        print("开始整合文件夹！")
        image_source_folder = "ExDark/ExDark"
        image_target_folder = "VOCdevkit/VOC2007/JPEGImages"
        extract_images(image_source_folder, image_target_folder)
        Annno_source_folder = "ExDark_Annno/ExDark_Annno"
        Annno_target_folder = "VOCdevkit/VOC2007/Annotations"
        extract_Annno(Annno_source_folder, Annno_target_folder)
        os.makedirs("VOCdevkit/VOC2007/ImageSets/Main")
        print("文件整合完毕")

    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")

    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        #递归获取所有txt文件
        def get_all_xml_files(directory):
            xml_files = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(".txt"):
                        xml_files.append(os.path.join(root, file))
            return xml_files
        xml_files = get_all_xml_files(xmlfilepath)
        total_xml = [os.path.splitext(os.path.basename(xml))[0] for xml in xml_files]

        ####################################################################################

        saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        num     = len(total_xml)  
        list    = range(num)  
        tv      = int(num*trainval_percent)  
        tr      = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train   = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("train size",tr)
        ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        
        for i in list:  
            name=total_xml[i][:]+'\n'
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)  
        
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/VOC%s/JPEGImages/%s'%(os.path.abspath(VOCdevkit_path), year, image_id))    #绝对路径
                #list_file.write('%s/VOC%s/JPEGImages/%s'%(os.path.join('./VOCdevkit'), year, image_id))        #相对路径
                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")
        
        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()

        str_nums = [str(int(x)) for x in nums]
        tableData = [
            classes, str_nums
        ]
        colWidths = [0]*len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("（重要的事情说三遍）。")
