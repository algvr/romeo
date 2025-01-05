import collections
import json
import numpy as np
import os
# https://pymeshlab.readthedocs.io/en/latest/filter_list.html
# import pymeshlab
from scipy.spatial import KDTree


colors_array = np.asarray([[0, 0, 1],
                           [0, 1, 0],
                           [1, 0, 0],
                           [0.5, 0.5, 0],
                           [0., 0.5, 0.5],
                           [0.5, 0., 0.5],
                           [0.6, 0.3, 0],
                           [0.6, 0., 0.3],
                           [0., 0.6, 0.3],
                           [0.3, 0.6, 0.],
                           [0.3, 0., 0.6],
                           [0.0, 0.3, 0.6]])


def simplify_obj(nf, orig_ply_path, output_obj_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(orig_ply_path)

    ms.generate_simplified_point_cloud(samplenum=nf)
    ms.compute_normal_for_point_clouds()

    # ms.generate_surface_reconstruction_screened_poisson()
    ms.generate_surface_reconstruction_ball_pivoting()
    ms.meshing_close_holes()
    # ms.generate_marching_cubes_apss()
    # tmp_obj_path = "./tmp_object.obj"
    # ms.save_current_mesh(tmp_obj_path)

    # ms = pymeshlab.MeshSet()
    # ms.load_new_mesh(tmp_obj_path)
    # ms.meshing_decimation_quadric_edge_collapse(targetfacenum=nf, preserveboundary=True, planarquadric=True)
    ms.compute_selection_by_non_manifold_edges_per_face()
    ms.compute_selection_by_non_manifold_per_vertex()
    ms.meshing_remove_selected_vertices_and_faces()
    ms.meshing_close_holes()
    # ms.meshing_re_orient_faces_coherentely()
    # m = ms.current_mesh()
    # print(ms.number_meshes())
    # print(m.vertex_number())
    # ms.meshing_invert_face_orientation()
    ms.save_current_mesh(output_obj_path)
    print("save object to ", output_obj_path)
    # os.remove(tmp_obj_path)


def load_label_ids(label_file_path):
    with open(label_file_path) as f:
        contents = f.readlines()
    labels = [int(l.rstrip()) for l in contents]
    # print(labels)
    return labels


def load_obj(filename_obj, normalization=False):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    vertex_colors = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
            if len(line.split()) > 4:
                vertex_colors.append([float(v) for v in line.split()[4:]])
    vertices = np.vstack(vertices).astype(np.float32)
    if len(vertex_colors):
        vertex_colors = np.vstack(vertex_colors).astype(np.float32)

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype(np.int32) - 1

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= np.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    if len(vertex_colors):
        return vertices, faces, vertex_colors
    else:
        return vertices, faces


def load_pts(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        pts = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] for line in lines],
                       dtype=np.float32)
        colors = np.array([[float(line.split()[3]), float(line.split()[4]), float(line.split()[5])] for line in lines],
                       dtype=np.float32)
        return pts, colors


def find_obj_labels(vertices, points, labels):
    new_lables = []
    tree = KDTree(points)
    for v in vertices:
        dist, idx = tree.query(v, k=1)
        new_lables.append(labels[idx])
    return new_lables


def find_first_expanding_level(node_level):
    target_level = 0
    if len(node_level[1]) > 1:
        target_level = 1
        target_set = node_level[target_level]
    elif len(node_level[2]) > 1:
        target_level = 2
        target_set = node_level[target_level]
    elif len(node_level[3]) > 1:
        target_level = 3
        target_set = node_level[target_level]

    return target_level, target_set


def map_leaf_index_to_target_index(all_nodes, target_set, label_set):
    target_l = 0
    map_pairs = []
    for nl in all_nodes:
        if nl in target_set:
            target_l = nl
        if nl in label_set:
            map_pairs.append([nl, target_l])
    # print(map_pairs)
    return map_pairs


def detect_parts_with_same_names(target_labels):
    part_labels = list(target_labels.values())
    part_keys = list(target_labels.keys())
    redundant_l = [item for item, count in collections.Counter(part_labels).items() if count > 1]
    redundant_pairs = []
    for rl in redundant_l:
        ind_pairs = np.where(rl == np.asarray(part_labels))[0]
        k_pairs = []
        for i in ind_pairs:
            k_pairs.append(part_keys[i])
        redundant_pairs.append(k_pairs)
    return redundant_pairs


def map_leaf_to_top_level(partnet_obj_dir, new_labels):

    result_file = os.path.join(partnet_obj_dir, 'result.json')
    with open(result_file, "r") as f:
        tree_hier = json.load(f)[0]

    node_level = {}
    node_loc = {}
    all_nodes = []
    node_labels = {}

    def find_level_loc(cur_tree_hier, cur_level, cur_loc):
        node_id = cur_tree_hier['id']
        all_nodes.append(node_id)

        if 'children' in cur_tree_hier:
            child_nodes = cur_tree_hier['children']
        else:
            child_nodes = []

        if cur_level not in node_level.keys():
            node_level[cur_level] = []
            node_labels[cur_level] = []

        node_level[cur_level].append(node_id)
        node_labels[cur_level].append(cur_tree_hier['text'])
        if len(child_nodes) == 0:
            return 1
        else:
            old_cur_loc = cur_loc
            for child_node in child_nodes:
                child_loc = find_level_loc(child_node, cur_level+1, cur_loc)
                node_loc[child_node['id']] = cur_loc
                # node_labels[child_node['id']] = child_node['name']
                cur_loc += child_loc + 1
            return cur_loc - old_cur_loc

    root_node = tree_hier['id']
    node_loc[root_node] = 0
    find_level_loc(tree_hier, 0, 0)
    # print(node_level)
    # print(node_loc)
    # print(all_nodes)
    # print(node_labels)

    # find our target hierarchical level and indices
    target_level, target_set = find_first_expanding_level(node_level)
    label_set = list(set(new_labels))

    # map_pairs [leaf_idx, target_idx (parent idx)]
    map_pairs = map_leaf_index_to_target_index(all_nodes, target_set, label_set)

    # get semantic labels for target set
    target_labels = {}
    for i in range(len(target_set)):
        target_labels[target_set[i]] = node_labels[target_level][i]
    # print(target_labels)

    # map leaf indice to semantic labels of parent node
    mapped_labels = {}
    for p in map_pairs:
        mapped_labels[p[0]] = target_labels[p[1]]
    # print(mapped_labels)

    # detect parts that have the same semantic labels
    redundant_pairs = detect_parts_with_same_names(target_labels)
    # print(redundant_pairs)

    pairs_with_the_same_label = []
    map_pairs = np.asarray(map_pairs)
    for sp in redundant_pairs:
        p = []
        for e in sp:
            v = map_pairs[np.where(map_pairs[:, 1] == e)[0], 0]
            p.append(v)
        pairs_with_the_same_label.append(p)
    # print(pairs_with_the_same_label)

    return mapped_labels, pairs_with_the_same_label


def generate_json(vertices, label_indices, semantic_labels, pairs_with_the_same_label,
                  output_file_path):
    label_indices = np.asarray(label_indices)
    # print(semantic_labels)

    # give redundant labels different name (separate them as left and right)
    for p in pairs_with_the_same_label: # assuming only two parts have the same name
        idx0 = np.where(label_indices == p[0])[0]
        v0 = np.mean(vertices[idx0, :], axis=0)
        idx1 = np.where(label_indices == p[1])[0]
        v1 = np.mean(vertices[idx1, :], axis=0)
        # print(v0)
        # print(v1)
        # print(p[0])

        ol = semantic_labels[p[0][0]]
        ol2 = semantic_labels[p[1][0]]
        assert ol2 == ol

        if v0[0] > v1[0]:
            semantic_labels[p[0][0]] = ol + " Left"
            semantic_labels[p[1][0]] = ol + " Right"
        elif v0[0] < v1[0]:
            semantic_labels[p[0][0]] = ol + " Right"
            semantic_labels[p[1][0]] = ol + " Left"
    # print(semantic_labels)

    # write indices into a json file
    res = {}
    for k in semantic_labels.keys():
        semantic_label = semantic_labels[k]
        indices = np.where(label_indices == k)[0]
        if semantic_label in res.keys(): # parts that have the same parent node
            res[semantic_label] += indices.tolist()
        else:
            res[semantic_label] = indices.tolist()

    with open(output_file_path, 'w') as fout:
        json.dump(res, fout, indent=4)


def export_ply_from_json_for_part(out, vertices, json_file_path, part_label):
    with open(json_file_path, 'r') as handle:
        parsed = json.load(handle)

    label_vert_indices = parsed[part_label]

    with open(out, 'w') as fout:
        fout.write('ply\n')
        fout.write('format ascii 1.0\n')
        fout.write('element vertex ' + str(vertices.shape[0]) + '\n')
        fout.write('property float x\n')
        fout.write('property float y\n')
        fout.write('property float z\n')
        fout.write('property uchar red\n')
        fout.write('property uchar green\n')
        fout.write('property uchar blue\n')
        fout.write('end_header\n')

        for i in range(vertices.shape[0]):
            if i in label_vert_indices:
                cur_color = [0.5, 0., 0.5] #[139, 90, 0]
            else:
                cur_color = [0., 1., 0.]# [255, 255, 255]
            fout.write('%f %f %f %d %d %d\n' % (vertices[i, 0], vertices[i, 1], vertices[i, 2],
                                                int(cur_color[0] * 255), int(cur_color[1] * 255),
                                                int(cur_color[2] * 255)))


def export_ply_from_json(out, vertices, json_file_path):
    num_colors = len(colors_array)

    with open(json_file_path, 'r') as handle:
        parsed = json.load(handle)

    with open(out, 'w') as fout:
        fout.write('ply\n')
        fout.write('format ascii 1.0\n')
        fout.write('element vertex ' + str(vertices.shape[0]) + '\n')
        fout.write('property float x\n')
        fout.write('property float y\n')
        fout.write('property float z\n')
        fout.write('property uchar red\n')
        fout.write('property uchar green\n')
        fout.write('property uchar blue\n')
        fout.write('end_header\n')

        c_idx = 0
        for k in parsed.keys():
            vert_indices = parsed[k]
            cur_color = colors_array[c_idx % num_colors]
            for v in vert_indices:
                fout.write('%f %f %f %d %d %d\n' % (vertices[v, 0], vertices[v, 1], vertices[v, 2],
                                                    int(cur_color[0] * 255), int(cur_color[1] * 255),
                                                    int(cur_color[2] * 255)))
            c_idx += 1


def main(obj_ids, cat, nf):

    base_path = "/mnt/scratch/xiwang/datasets/PartNet/data_v0"
    output_base_path = os.path.join("/local/home/xiwang1/projects/bointeraction/models/meshes/", cat)
    if not os.path.exists(output_base_path):
        os.mkdir(output_base_path)

    for obj_id in obj_ids:
        # simplify original object mesh and save in obj
        partnet_obj_dir = os.path.join(base_path, obj_id)
        partnet_base_path = os.path.join(partnet_obj_dir, 'point_sample')
        orig_ply_path = os.path.join(partnet_base_path, "ply-10000.ply")
        output_obj_path = os.path.join(output_base_path, '{}_{}_{}.obj'.format(cat, obj_id, nf))
        simplify_obj(nf, orig_ply_path, output_obj_path)

        # load original labels
        label_file_path = os.path.join(partnet_base_path, 'label-10000.txt')
        orig_labels = load_label_ids(label_file_path)  # leaf level labels of vertex [0, ..., N]

        # nearest neighbour based label search
        vertices, _, obj_colors = load_obj(output_obj_path)
        # vertices, _ = load_obj(output_obj_path)
        # load original pts file
        pts_file_path = os.path.join(partnet_base_path, "pts-10000.pts")
        points, _ = load_pts(pts_file_path)

        labels = find_obj_labels(vertices, points, orig_labels)
        # print(labels)
        # print(len(labels))

        # generate leaf level semantic labels to first expanding level
        label_dict, pairs_with_the_same_label = map_leaf_to_top_level(partnet_obj_dir, labels)

        output_json_path = os.path.join(output_base_path, '{}_{}_{}_labels.json'.format(cat, obj_id, nf))
        generate_json(vertices, labels, label_dict, pairs_with_the_same_label, output_json_path)

        # # check label correctness
        # output_tmp_ply_path = os.path.join(output_base_path, '{}_{}_check.ply'.format(cat, obj_id))
        # export_ply_from_json(output_tmp_ply_path, vertices, output_json_path)
        # export_ply_from_json_for_part(output_tmp_ply_path, vertices, output_json_path, "Chair Arm Left")


def reconstruct_obj(orig_ply_path, output_obj_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(orig_ply_path)
    ms.compute_normal_for_point_clouds()
    ms.generate_surface_reconstruction_ball_pivoting()
    ms.meshing_close_holes()
    ms.meshing_invert_face_orientation()
    ms.save_current_mesh(output_obj_path)
    print("save object to ", output_obj_path)


def extract_labels(obj_ids, cat):
    base_path = "/mnt/scratch/xiwang/datasets/PartNet/data_v0"
    output_base_path = os.path.join("/local/home/xiwang1/projects/bointeraction/models/meshes/", cat)
    if not os.path.exists(output_base_path):
        os.mkdir(output_base_path)

    nf = 10000

    for obj_id in obj_ids:
        partnet_obj_dir = os.path.join(base_path, obj_id)
        partnet_base_path = os.path.join(partnet_obj_dir, 'point_sample')
        orig_ply_path = os.path.join(partnet_base_path, "ply-10000.ply")
        output_obj_path = os.path.join(output_base_path, '{}_{}_{}.obj'.format(cat, obj_id, nf))
        reconstruct_obj(orig_ply_path, output_obj_path)
        vertices, _, _ = load_obj(output_obj_path)

        # load original labels
        label_file_path = os.path.join(partnet_base_path, 'label-10000.txt')
        orig_labels = load_label_ids(label_file_path)  # leaf level labels of vertex [0, ..., N]

        # generate leaf level semantic labels to first expanding level
        label_dict, pairs_with_the_same_label = map_leaf_to_top_level(partnet_obj_dir, orig_labels)

        output_json_path = os.path.join(output_base_path, '{}_{}_{}_labels.json'.format(cat, obj_id, nf))
        generate_json(vertices, orig_labels, label_dict, pairs_with_the_same_label, output_json_path)


def list_semintic_lables(obj_ids, cat):
    base_path = os.path.join("/local/home/xiwang1/projects/bointeraction/models/meshes/", cat)
    labels = []
    for obj_id in obj_ids:
        json_file_path = os.path.join(base_path, '{}_{}_1000_labels.json'.format(cat, obj_id))
        with open(json_file_path, 'r') as handle:
            d = json.load(handle)
        labels += list(d.keys())

    print(set(labels))
    output_file_path = os.path.join("/local/home/xiwang1/projects/bointeraction/models/meshes/", '{}_labels.json'.format(cat))
    d = {}
    d['labels'] = list(set(labels))
    with open(output_file_path, 'w') as fout:
        json.dump(d, fout, indent=4)


def read_cluster_ids(cat):
    txt_file_path = os.path.join('../partnet/cluster/{}'.format(cat), 'center_obj.txt')
    with open(txt_file_path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


if __name__ == '__main__':
    # list_of_chairs = ['44236', '36547', '35151', '37846', '35501', '36917', '39539',
    #                   '37276', '2364', '40675', '39247', '37661']
    # list_of_chairs = ['36547']
    list_of_chairs = read_cluster_ids('chair')

    nf = 1000  # simplify obj to target face number
    main(list_of_chairs, "chair", nf)
    nf = 10000
    extract_labels(list_of_chairs, 'chair')

    list_semintic_lables(list_of_chairs, 'chair')
