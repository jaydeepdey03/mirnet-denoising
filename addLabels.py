import os
import csv
import xml.etree.ElementTree as ET

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    image_path = root.find('path').text  # Use the path tag to get the image file path
    image_filename = os.path.basename(image_path)  # Extract the image filename from the path

    # Verify that the filename in the XML matches the actual image filename
    assert filename == image_filename, f"Filename mismatch for {xml_file}"

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    objects = root.findall('object')
    data = []
    for obj in objects:
        label = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        data.append([filename, width, height, label, xmin, ymin, xmax, ymax])

    return data

def convert_to_csv(xml_folder, csv_file):
    data_list = []

    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_folder, xml_file)
            data_list.extend(parse_xml(xml_path))

    # Sort the data based on the 'filename' column
    data_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x[0]))))


    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        csv_writer.writerows(data_list)

# Example usage
xml_folder = './dataset_polarimetric/train/LABELS'
csv_file = './dataset_polarimetric/train/labels.csv'
convert_to_csv(xml_folder, csv_file)
