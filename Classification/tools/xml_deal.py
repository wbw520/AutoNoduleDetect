from xml.dom import minidom
import os


class MakeXml():
    def __init__(self, sizes, OBs, paths, file_names, xml_names):
        self.dom = minidom.Document()
        self.size = sizes
        self.object = OBs
        self.xml_name = xml_names
        self.path = paths
        self.file_name = file_names

    def make_construct(self, module_name, text, father):
        child_node = self.dom.createElement(module_name)
        father.appendChild(child_node)
        if text != "":
            folder_text = self.dom.createTextNode(text)
            child_node.appendChild(folder_text)
        return child_node

    def xml(self):
        root_node = self.dom.createElement("annotation")
        self.dom.appendChild(root_node)
        self.make_construct("folder", "Desktop", root_node)
        self.make_construct("filename", self.file_name, root_node)
        self.make_construct("path", self.path, root_node)

        source_node = self.make_construct("source", "", root_node)
        self.make_construct("database", "Osaka_HP", source_node)

        size_node = self.make_construct("size", "", root_node)
        hehe = ["width", "height", "depth"]
        for i in range(len(hehe)):
            self.make_construct(hehe[i], str(self.size[i]), size_node)

        segmented_node = self.make_construct("segmented", "0", root_node)

        prepare_name = locals()
        for i in range(len(self.object)):
            prepare_name["object_node"+str(i)] = self.make_construct("object", "", root_node)
            self.make_construct("name", self.object[i][4], prepare_name["object_node"+str(i)])
            self.make_construct("pose", "Unspecified", prepare_name["object_node"+str(i)])
            self.make_construct("truncated", "0", prepare_name["object_node"+str(i)])
            self.make_construct("difficult", "0", prepare_name["object_node"+str(i)])

            prepare_name["bndbox_node" + str(i)] = self.make_construct("bndbox", "", prepare_name["object_node"+str(i)])
            coordinate = ["xmin", "ymin", "xmax", "ymax"]
            for j in range(4):
                self.make_construct(coordinate[j], str(self.object[i][j]), prepare_name["bndbox_node" + str(i)])

        with open(self.xml_name+".xml", "w", encoding="UTF-8") as fh:
            self.dom.writexml(fh, indent="", addindent="\t", newl="\n", encoding="UTF-8")
            print(self.xml_name+":xml ok")


def get2(root):
    for root, dirs, SB in os.walk(root):
        return SB


def read_coordinate(name, root):
    with open(root + name + ".txt", "r", encoding="UTF-8") as data:
        object = []
        lines = data.readlines()
        for line in lines:
            wbw = line.split(",")
            a, b, c, d = wbw[0], wbw[1], wbw[2], wbw[3][:-1]
            object.append([a, b, c, d, "nodule"])
        return object


if __name__ == '__main__':
    root = "/"  # root of all txt file record coordinate
    root_image = "/"  # root of image
    root_for_xml = "/"  # root for save traslated xml file
    all_txt = get2(root)
    for i in range(len(all_txt)):
        name = all_txt[i][:-4]
        size = [1000, 1000, 1]
        object = read_coordinate(name, root)
        xml_name = root_for_xml + name
        file_name = name+".jpg"
        path = root_image + name + ".jpg"
        MakeXml(size, object, path, file_name, xml_name).xml()