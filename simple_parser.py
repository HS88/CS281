
import sys
sys.path.insert(0, '..')
import __init__
def attach_classes(vsrl_data, coco):
  """
  def vsrl_data = attach_gt_boxes(vsrl_data, coco)
  """
  anns = coco.loadAnns(vsrl_data['ann_id'].ravel().tolist());
  vsrl_data['category_id'] = \
      np.nan * np.zeros((vsrl_data['role_object_id'].shape[0], \
                          vsrl_data['role_object_id'].shape[1]), dtype=np.int)
  for i in range(vsrl_data['role_object_id'].shape[1]):
    has_role = np.where(vsrl_data['role_object_id'][:,i] > 0)[0]
    if has_role.size > 0:
      anns = coco.loadAnns(vsrl_data['role_object_id'][has_role, i].ravel().tolist());
      cat = np.vstack([a['category_id'] for a in anns])
      vsrl_data['category_id'][has_role, i:(i+1)] = cat;
  return vsrl_data




def get_object_box_coords(class_name, all_imgs, filename, human_bbox, action, action_id, rois):
  #given class_name, all_imgs, filename, and rois, appends the appropriate bboxes and triples to all_imgs, and returns
  for i in range(rois.shape[0]):
        roi = rois[i,:].astype(np.int)
        x1 = roi[0]
        y1 = roi[1]
        x2 = roi[2]
        y2 = roi[3]

        object_bbox = {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)}
        all_imgs[filename]['bboxes'].append(object_bbox)
        all_imgs[filename]['triples'].append((human_bbox, {"action": action_id}, object_bbox))
  return all_imgs

def get_data_set(coco, all_imgs, classes_count, actions_count, input_path, imset):
    #input path is full path
    #given starting action_map, imset and input_path, returns:
    #-- action_mapping maps actions to ints
    #-- all_imgs in imset
    #-- classes_count: classes and their count, with initial starting classes count
    #--actions_count: actions and their count, with initial starting actions count
    #imset can only be train, trainval, val, or test
    #only encodes images with at least one object
    #each image gets an arbituarily picked correct human, action, object tuple

    # Load the VCOCO annotations for vcoco_train image set
    vcoco_all = vu.load_vcoco('vcoco_' + imset)
    output_path = "/data/home/harmeetsingh/vcoco"

    from PIL import Image
    import urllib, cStringIO

    for a in range(len(vcoco_all)):
        x = vcoco_all[a]
        vcoco = vu.attach_gt_boxes(x, coco)
        vcoco = attach_classes(vcoco, coco)
        #load only images with object bounding boxes
        positive_index = np.where(vcoco['label'] == 1)[0]

        for i in range(len(positive_index)):
            id = positive_index[i]
            coco_image = coco.loadImgs(ids=[vcoco['image_id'][id][0]])[0]
            filename = input_path + "/" + str(vcoco['image_id'][id][0]) + ".jpg"

            #figure out if has an object bbox or not
            role_bbox = vcoco['role_bbox'][id, :] * 1.
            role_bbox = role_bbox.reshape((-1, 4))
            is_object = False
            for k in range(1, len(vcoco['role_name'])):
                if not np.isnan(role_bbox[k, 0]):
                    is_object = True
                    break
            if filename not in all_imgs and is_object:
                all_imgs[filename] = {}

                #file = cStringIO.StringIO(urllib.urlopen(filename).read())
                img = np.asarray(Image.open(filename))
                (rows,cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = output_path + "/" + str(vcoco['image_id'][id][0]) + ".jpg"
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                all_imgs[filename]['imageset'] = imset

  
                #parse bounding box for agent
                rois = vcoco['bbox'][[id],:]
                roi = rois[0,:].astype(np.int)
                x1 = roi[0]
                y1 = roi[1]
                x2 = roi[2]
                y2 = roi[3]
                human_bbox_dict = {'class': 'person', 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)}
                all_imgs[filename]['bboxes'] = [human_bbox_dict]
                classes_count['person'] = classes_count['person'] + 1

                # agent, verb, object triples
                all_imgs[filename]['triples'] = []

                role_bbox = vcoco['role_bbox'][id,:]*1.
                role_bbox = role_bbox.reshape((-1,4))
                encoded_triplet = False
                for j in range(1, len(vcoco['role_name'])):
                    if not encoded_triplet and not np.isnan(role_bbox[j,0]):
                        category_id = vcoco['category_id'][id][j]
                        cats = coco.loadCats(ids = [category_id])[0]
                        all_imgs = get_object_box_coords(cats['name'], all_imgs, filename, human_bbox_dict, vcoco['action_name'],a, role_bbox[[j],:])
                        classes_count[cats['name']] = classes_count[cats['name']] + 1
                        actions_count[vcoco['action_name']] = actions_count[vcoco['action_name']] + 1
                        encoded_triplet = True
    return all_imgs, classes_count, actions_count

def get_data(input_path):#may need change to get_data_vcoco #TODO
  #input_path is full path
  # parses data into
  #--action_mapping (maps actions to ints)
  #-- all_data(holds the data)
  #-- classes_count(dictionary mapping classes to its count)
  #-- class_mapping(dictionary mapping classes to its numerical value)
  #-- actions_count(dictionary mapping actions to its count)
  #-- action_mapping(dictionary mapping actions to its numerical value)

  # Load COCO annotations for V-COCO images
  coco = vu.load_coco()

  # assuming same format for each image set
  # Load the VCOCO annotations for vcoco_train image set
  vcoco_all = vu.load_vcoco('vcoco_val')

  #get action_mapping and intialize actions_count
  action_mapping = {'bg': len(vcoco_all)}
  actions_count = {'bg':0}
  #assuming same format for each image set
  for a in range(len(vcoco_all)):
      action_mapping[vcoco_all[a]['action_name']] = a
      actions_count[vcoco_all[a]['action_name']] = 0

  #get class_mapping and initialize classes_count
  cat_ids = coco.getCatIds()
  cats = coco.loadCats(cat_ids)
  class_mapping = {'bg': len(cats)}

  classes_count = {'bg': 0}
  for c in range(len(cats)):
      cat = cats[c]
      class_mapping[cat['name']] = c
      classes_count[cat['name']] = 0

  #get all data
  print "transforming train data"
  all_imgs, classes_count, actions_count = get_data_set(coco, {}, classes_count, actions_count, input_path, "train")
  print "transforming val data"
  all_imgs, classes_count, actions_count = get_data_set(coco, all_imgs, classes_count, actions_count, input_path, "val")
  print "transform test data"
  all_imgs, classes_count, actions_count = get_data_set(coco, all_imgs, classes_count, actions_count, input_path, "test")
  all_data = []
  for key in all_imgs:
      all_data.append(all_imgs[key])



 # print actions_count
  print action_mapping
  #get classes_count
  #classes_count = {"object":neg_count1 + neg_count2 + neg_count3, "human":pos_count1 + pos_count2 + pos_count3}

  return all_data, classes_count, class_mapping, actions_count, action_mapping
