import cv2, shapefile, math
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, draw, data, color, morphology

def filling(img):
    img_contour = img.copy()
    img_contour[img_contour > 0] = 0
    contour_full = img_contour.copy()
    contours = measure.find_contours(img, 0)
    print(len(contours))
    for n, contour in enumerate(contours):
        # print(n)
        rr, cc = draw.polygon(contour[:, 0], contour[:, 1])
        contour_full[rr, cc] = 255
        if n % 1000 == 0:
            print(n)
    # 删除小块
    print("删除小块")
    contour_full_no = contour_full > 0
    contour_full_no = morphology.remove_small_objects(contour_full_no, min_size=200, connectivity=1)
    cv2.imwrite("F:\\regularize\\full.tif", contour_full_no.astype(int) * 255)
    print("空洞填充完毕")
    return contour_full_no.astype(int)*255


def shp(shp_img):
    w = shapefile.Writer(shapefile.POLYGON)
    w.autoBalance = 1
    w.field('FIRST_FLD', 'C', '40')
    img_contour = shp_img.copy()
    img_contour[img_contour > 0] = 0
    contours = measure.find_contours(shp_img, 0.5)
    print(len(contours))
    for n, contour in enumerate(contours):
        appr_img = contour.copy()
        for _ in range(10):
            appr_img = measure.subdivide_polygon(appr_img, degree=2)
        appr_img[:, [1, 0]] = appr_img[:, [0, 1]]
        appr_img[:, [1]] = 0 - appr_img[:, [1]]
        w.poly(parts=[appr_img.tolist()])
        w.record('')
        # contour[:, [1, 0]] = contour[:, [0, 1]]
        # contour[:, [1]] = 0 - contour[:, [1]]
        # w.poly(parts=[contour.tolist()])
        # w.record('')

    w.save('F:\\regularize\\full.shp')


def length(x1, y1, x2, y2):
    l = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))
    return l


def tan_angle(x1, y1, x2, y2):
    if x2-x1==0:
        a = 90
    else:
        s= (y2-y1)/(x2-x1)
        a = math.atan(s)*180/math.pi
        if a < 0:
            a = a+180
    return a


def angle_dif(angle1, angle2):
    dif = angle1 - angle2
    if abs(dif) > 90:
        jiajiao = 180-abs(dif)
    else:
        jiajiao = abs(dif)
    if jiajiao > 45:
        jiajiao = 90-jiajiao
        vert = 1
    else:
        vert = 0
    return dif, jiajiao, vert


def Rotate(line, dif, jiajiao, new_angle, prall_or_vert):
    midx = (line[2][0] - line[1][0]) / 2 + line[1][0]
    midy = (line[2][1] - line[1][1]) / 2 + line[1][1]

    if prall_or_vert == 1:
        jiajiao = 90-jiajiao

    if jiajiao < 15:
        #     旋转终点
        dif = dif / 180 * math.pi
        nex = line[2][0] - midx
        ney = line[2][1] - midy
        xnex = nex * math.cos(dif) - ney * math.sin(dif)
        xney = ney * math.cos(dif) + nex * math.sin(dif)
        line[2] = (xnex+midx, xney+midy)
        #     旋转起点
        nsx = line[1][0] - midx
        nsy = line[1][1] - midy
        xnsx = nsx * math.cos(dif) - nsy * math.sin(dif)
        xnsy = nsy * math.cos(dif) + nsx * math.sin(dif)
        line[1] = (xnsx+midx, xnsy+midy)
        if new_angle>180:
            new_angle = new_angle-180
        line[4] = new_angle
    if jiajiao > 75:
        #     旋转终点
        dif = (dif + 90) / 180 * math.pi
        nex = line[2][0] - midx
        ney = line[2][1] - midy
        xnex = nex * math.cos(dif) - ney * math.sin(dif)
        xney = ney * math.cos(dif) + nex * math.sin(dif)
        line[2] = (xnex+midx, xney+midy)
        #     旋转起点
        nsx = line[1][0] - midx
        nsy = line[1][1] - midy
        xnsx = nsx * math.cos(dif) - nsy * math.sin(dif)
        xnsy = nsy * math.cos(dif) + nsx * math.sin(dif)
        line[1] = (xnsx+midx, xnsy+midy)
        new_angle = new_angle + 90
        if new_angle > 180:
            new_angle = new_angle-180
        line[4] = new_angle
    return line


def short_Rotate(line, dif, jiajiao, new_angle, prall_or_vert):
    midx = (line[2][0] - line[1][0]) / 2 + line[1][0]
    midy = (line[2][1] - line[1][1]) / 2 + line[1][1]

    if prall_or_vert == 1:
        jiajiao = 90-jiajiao

    if jiajiao <= 45:
        #     旋转终点
        dif = dif / 180 * math.pi
        nex = line[2][0] - midx
        ney = line[2][1] - midy
        xnex = nex * math.cos(dif) - ney * math.sin(dif)
        xney = ney * math.cos(dif) + nex * math.sin(dif)
        line[2] = (xnex+midx, xney+midy)
        #     旋转起点
        nsx = line[1][0] - midx
        nsy = line[1][1] - midy
        xnsx = nsx * math.cos(dif) - nsy * math.sin(dif)
        xnsy = nsy * math.cos(dif) + nsx * math.sin(dif)
        line[1] = (xnsx+midx, xnsy+midy)
        if new_angle>180:
            new_angle = new_angle-180
        line[4] = new_angle
    if jiajiao > 45:
        #     旋转终点
        dif = (dif + 90) / 180 * math.pi
        nex = line[2][0] - midx
        ney = line[2][1] - midy
        xnex = nex * math.cos(dif) - ney * math.sin(dif)
        xney = ney * math.cos(dif) + nex * math.sin(dif)
        line[2] = (xnex+midx, xney+midy)
        #     旋转起点
        nsx = line[1][0] - midx
        nsy = line[1][1] - midy
        xnsx = nsx * math.cos(dif) - nsy * math.sin(dif)
        xnsy = nsy * math.cos(dif) + nsx * math.sin(dif)
        line[1] = (xnsx+midx, xnsy+midy)
        new_angle = new_angle + 90
        if new_angle > 180:
            new_angle = new_angle - 180
        line[4] = new_angle
    return line


def GeneralEquation(first_x, first_y, second_x, second_y):
    # 一般式 Ax+By+C=0
    A = second_y - first_y
    B = first_x - second_x
    C = second_x * first_y - first_x * second_y
    return A, B, C


def GetIntersectPointofLines(qd1, zd1, qd2, zd2):
    A1, B1, C1 = GeneralEquation(qd1[0], qd1[1], zd1[0], zd1[1])
    A2, B2, C2 = GeneralEquation(qd2[0], qd2[1], zd2[0], zd2[1])
    m = A1 * B2 - A2 * B1
    if m == 0:
        x = None
        y = None
        flg = False
    else:
        x = (C2 * B1 - C1 * B2) / m
        y = (C1 * A2 - C2 * A1) / m
        flg = True
    return x, y, flg


def dis_parr(qd1, zd1, qd2, zd2):
    # A1, B1, C1 = GeneralEquation(qd1[0], qd1[1], zd1[0], zd1[1])
    A2, B2, C2 = GeneralEquation(qd2[0], qd2[1], zd2[0], zd2[1])
    # dis = abs(C2-C1)/(np.sqrt(np.square(A1) + np.square(B1)))
    dis = abs(A2*(qd1[0]+zd1[0])/2+B2*(qd1[1]+zd1[1])/2+C2)/(np.sqrt(np.square(A2) + np.square(B2)))
    return dis


def Sort_number(lines):
    for i in range(len(lines)):
        lines[i][0] = i
    return lines


#  sn:序号，zd：终点，angle：角度
def add_vertical(sn, zd, angle):
    if angle != 90:
        angle = angle + 90
        k = math.tan(angle/180*math.pi)
        b = zd[1]-k*zd[0]
        ny = k*(zd[0]+1)+b
        nzd = (zd[0]+1, ny)
    else:
        nzd = (zd[0]+1, zd[1])
        angle = angle+90
    l = length(zd[0], zd[1], nzd[0], nzd[1])
    if angle > 180:
        angle = angle-180
    nline = [sn+1, zd, nzd, l, angle]
    return nline


def com_area(contour):
    n = len(contour)
    s = 0
    for i in range(n-1):
        s = s + contour[i][0]*contour[i+1][1]-contour[i+1][0]*contour[i][1]
    s = s + contour[n-1][0]*contour[0][1]-contour[0][0]*contour[n-1][1]
    s = math.fabs(s)/2
    return s


def find_nearst(qd1, zd1, qd2, zd2):
    d11 = abs(qd1[0] - qd2[0]) + abs(qd1[1] - qd2[1])
    d12 = abs(qd1[0] - zd2[0]) + abs(qd1[1] - zd2[1])
    d21 = abs(zd1[0] - qd2[0]) + abs(zd1[1] - qd2[1])
    d22 = abs(zd1[0] - zd2[0]) + abs(zd1[1] - zd2[1])
    d = [d11, d12, d21, d22]
    index = d.index(min(d))
    if index < 2:
        return qd1
    else:
        return zd1


def adjustment(initial, w, longT, prallT):
    # print(initial)
    lines = []  # 线段集合,0序号、1起点、2终点、3长度、4角度（与横轴的角度）
    for i in range(len(initial)-1):
        l = length(initial[i][0], initial[i][1], initial[i+1][0], initial[i+1][1])
        a = tan_angle(initial[i][0], initial[i][1], initial[i+1][0], initial[i+1][1])
        lines.append([i, initial[i], initial[i+1], l, a])
    l = length(initial[len(initial)-1][0], initial[len(initial)-1][1], initial[0][0], initial[0][1])
    a = tan_angle(initial[len(initial) - 1][0], initial[len(initial) - 1][1], initial[0][0], initial[0][1])
    lines.append([len(initial)-1, initial[len(initial)-1], initial[0], l, a])

    # fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    # ax0, ax1 = axes.ravel()
    # for line in lines:
    #     plt.plot([line[1][0], line[2][0]], [line[1][1], line[2][1]])
    # plt.show()

    # 获取所有的长度
    lengths = [x[3] for x in lines]
    longlines = []
    # 提取长边，默认阈值为6，如果最长边小于6，依次减少0.2m
    while len(longlines) == 0:
        longlines = [x[0] for x in lines if x[3] > longT]
        longT = longT-0.2
    shortlines = [x[0] for x in lines if x[3] <= longT]
    longst = lengths.index(max(lengths))
    main_dir = [lines[longst][4]]

#     主方向确定
    for longline in longlines:
        all_jj = []  # 与主方向夹角集合，用于判断
        for dir in main_dir:
            dif, jiajiao, _ = angle_dif(dir, lines[longline][4])
            all_jj.append(jiajiao)
        min_jiajiao = min(all_jj)
        if 15 < min_jiajiao < 75:
            main_dir.append(lines[longline][4])

#     调整长线
    for longline in longlines:
        all_jj = []  # 与主方向夹角集合，用于判断
        all_dif = []  #与主方向的角度集合，用于变换
        for dir in main_dir:
            dif, jiajiao, v = angle_dif(dir, lines[longline][4])
            all_jj.append(jiajiao)
            all_dif.append([dif, dir, v])
        min_jiajiao = min(all_jj)
        real_dif = all_dif[all_jj.index(min_jiajiao)][0]
        new_angle = all_dif[all_jj.index(min_jiajiao)][1]  # 新的角度
        prall_or_vert = all_dif[all_jj.index(min_jiajiao)][2]  # 是否垂直，0平行，1垂直
        newline = Rotate(lines[longline], real_dif, min_jiajiao, new_angle, prall_or_vert)
        lines[longline] = newline

#     调整短线
    for shortline in shortlines:
        all_jj = []
        all_dif = []
        for dir in main_dir:
            dif, jiajiao, v = angle_dif(dir, lines[shortline][4])
            all_jj.append(jiajiao)
            all_dif.append([dif, dir, v])
        min_jiajiao = min(all_jj)
        real_dif = all_dif[all_jj.index(min_jiajiao)][0]
        new_angle = all_dif[all_jj.index(min_jiajiao)][1]  # 新的角度
        prall_or_vert = all_dif[all_jj.index(min_jiajiao)][2]  # 是否垂直，0平行，1垂直
        newline = short_Rotate(lines[shortline], real_dif, min_jiajiao, new_angle, prall_or_vert)
        # newline[4] = all_dif[all_jj.index(min_jiajiao)][1]  # 新的角度
        lines[shortline] = newline

    # for line in lines:
    #     ax0.plot([line[1][0], line[2][0]], [line[1][1], line[2][1]])
    #     ax0.plot(line[1][0], line[1][1], '*')

#     剔除冗余平行线，或在错误地方加入短线
    nnode = 0
    nline = len(lines)  # 线段个数
    while nnode < nline-1:
        # print(nnode, nline)
        # print(abs(lines[nnode][4] - lines[nnode + 1][4]))
        if 5 < abs(lines[nnode][4] - lines[nnode+1][4]) < 175:
            nnode = nnode+1
        else:
            #   计算平行线的距离
            prall_dis = dis_parr(lines[nnode][1], lines[nnode][2], lines[nnode+1][1], lines[nnode+1][2])
            # print(prall_dis)
            if prall_dis > prallT:
                dian = find_nearst(lines[nnode][1], lines[nnode][2], lines[nnode+1][1], lines[nnode+1][2])
                add_line = add_vertical(lines[nnode][0], dian, lines[nnode][4])
                lines.insert(lines[nnode][0]+1, add_line)
                lines = Sort_number(lines)
                nnode = nnode + 1
                nline = nline + 1
            else:
                li = length(lines[nnode][1][0], lines[nnode][1][1], lines[nnode][2][0], lines[nnode][2][1])
                li2 = length(lines[nnode+1][1][0], lines[nnode+1][1][1], lines[nnode+1][2][0], lines[nnode+1][2][1])
                if li >= li2:
                    lines.remove(lines[nnode+1])
                else:
                    lines.remove(lines[nnode])
                lines = Sort_number(lines)
                nline = nline - 1
    # for line in lines:
    #     ax1.plot([line[1][0], line[2][0]], [line[1][1], line[2][1]])
    # plt.show()
    # 判断第一条边和最后一条边
    if abs(lines[-1][4] - lines[0][4]) < 5 or abs(lines[-1][4] - lines[0][4]) > 175:
        prall_dis = dis_parr(lines[-1][1], lines[-1][2], lines[0][1], lines[0][2])
        if prall_dis > prallT:
            dian = find_nearst(lines[-1][1], lines[-1][2], lines[0][1], lines[0][2])
            add_line = add_vertical(lines[-1][0], dian, lines[-1][4])
            lines.insert(lines[-1][0]+1, add_line)
            lines = Sort_number(lines)
        else:
            li = length(lines[-1][1][0], lines[-1][1][1], lines[-1][2][0], lines[-1][2][1])
            li2 = length(lines[0][1][0], lines[0][1][1], lines[0][2][0], lines[0][2][1])
            if li >= li2:
                lines.remove(lines[0])
            else:
                lines.remove(lines[-1])
            lines = Sort_number(lines)

    # for line in lines:
    #     plt.plot([line[1][0], line[2][0]], [line[1][1], line[2][1]])

    nodes = []
    inter_num = 0
    inter_n = len(lines)
    while inter_num < inter_n-1:
        x, y, flag = GetIntersectPointofLines(lines[inter_num][1], lines[inter_num][2], lines[inter_num + 1][1], lines[inter_num + 1][2])
        # print(flag)
        if flag==False:
            lines.remove(lines[inter_num+1])
            inter_n = inter_n-1
        else:
            nodes.append((x, y))
            inter_num = inter_num+1
    x, y, _ = GetIntersectPointofLines(lines[0][1], lines[0][2], lines[len(lines) - 1][1], lines[len(lines) - 1][2])
    nodes.append((x, y))

    if len(nodes) > 3:
        w.poly(parts=[nodes])
        w.record(len(nodes))

    # for i in range(len(nodes)-1):
    #     plt.plot([nodes[i][0], nodes[i+1][0]], [nodes[i][1], nodes[i+1][1]])
    #     plt.plot(nodes[i][0], nodes[i][1], "*")
    # plt.plot([nodes[0][0], nodes[len(nodes)-1][0]], [nodes[0][1], nodes[len(nodes)-1][1]])
    # plt.show()


def group():
    s = shapefile.Reader('E:\\AerialImageDataset\\austin\\result\\last.shp')
    shapes = s.shapes()
    w = shapefile.Writer(shapefile.POLYGON)
    w.autoBalance = 1
    w.field('FIRST_FLD', 'C', '40')
    for shapenum in range(len(shapes)):
        print(shapenum)
        initial = shapes[shapenum].points[:-1]
        area = com_area(initial)
        # if area < 2000:
        #     longT = 6
        #     prallT = 0.8
        #     adjustment(initial, w, longT, prallT)
        # elif 2000 <= area < 5000:
        #     longT = 8
        #     prallT = 1.2
        #     adjustment(initial, w, longT, prallT)
        # elif 5000 <= area:
        #     longT = 12
        #     prallT = 1.6
        #     adjustment(initial, w, longT, prallT)
        if area < 80:
            longT = 6
            prallT = 0.8
            adjustment(initial, w, longT, prallT)
        elif 80 <= area < 200:
            longT = 8
            prallT = 1
            adjustment(initial, w, longT, prallT)
        elif 200 <= area:
            longT = 10
            prallT = 1.2
            adjustment(initial, w, longT, prallT)
    w.save('E:\\AerialImageDataset\\austin\\result\\new2.shp')
'''
def adjustment():
    s = shapefile.Reader('F:\\regularize\\last.tif')
    shapes = s.shapes()
    w = shapefile.Writer(shapefile.POLYGON)
    w.autoBalance = 1
    w.field('FIRST_FLD', 'C', '40')
    for shapenum in range(len(shapes)):
        print(shapenum)
        initial = shapes[shapenum].points[:-1]
        # print(initial)
        lines = []  # 线段集合,0序号、1起点、2终点、3长度、4角度（与横轴的角度）
        for i in range(len(initial)-1):
            l = length(initial[i][0], initial[i][1], initial[i+1][0], initial[i+1][1])
            a = tan_angle(initial[i][0], initial[i][1], initial[i+1][0], initial[i+1][1])
            lines.append([i, initial[i], initial[i+1], l, a])
        l = length(initial[len(initial)-1][0], initial[len(initial)-1][1], initial[0][0], initial[0][1])
        a = tan_angle(initial[len(initial) - 1][0], initial[len(initial) - 1][1], initial[0][0], initial[0][1])
        lines.append([len(initial)-1, initial[len(initial)-1], initial[0], l, a])

        # for line in lines:
        #     plt.plot([line[1][0], line[2][0]], [line[1][1], line[2][1]])
        # plt.show()
        longT = 6
        # 获取所有的长度
        lengths = [x[3] for x in lines]
        longlines = []
        # 提取长边，默认阈值为6，如果最长边小于6，依次减少0.5m
        while len(longlines) == 0:
            longlines = [x[0] for x in lines if x[3] > longT]
            longT = longT-0.5
        shortlines = [x[0] for x in lines if x[3] <= longT]
        longst = lengths.index(max(lengths))
        main_dir = [lines[longst][4]]
    #     主方向确定
        for longline in longlines:
            all_jj = []  # 与主方向夹角集合，用于判断
            for dir in main_dir:
                dif, jiajiao = angle_dif(dir, lines[longline][4])
                all_jj.append(jiajiao)
            min_jiajiao = min(all_jj)
            if 15 < min_jiajiao < 75:
                main_dir.append(lines[longline][4])

    #     调整长线
        for longline in longlines:
            all_jj = []  # 与主方向夹角集合，用于判断
            all_dif = []  #与主方向的角度集合，用于变换
            for dir in main_dir:
                dif, jiajiao = angle_dif(dir, lines[longline][4])
                all_jj.append(jiajiao)
                all_dif.append([dif, dir])
            min_jiajiao = min(all_jj)
            real_dif = all_dif[all_jj.index(min_jiajiao)][0]
            new_angle = all_dif[all_jj.index(min_jiajiao)][1]  # 新的角度
            newline = Rotate(lines[longline], real_dif, min_jiajiao, new_angle)
            lines[longline] = newline

    #     调整短线
        for shortline in shortlines:
            all_jj = []
            all_dif = []
            for dir in main_dir:
                dif, jiajiao = angle_dif(dir, lines[shortline][4])
                all_jj.append(jiajiao)
                all_dif.append([dif, dir])
            min_jiajiao = min(all_jj)
            real_dif = all_dif[all_jj.index(min_jiajiao)][0]
            new_angle = all_dif[all_jj.index(min_jiajiao)][1]  # 新的角度
            newline = short_Rotate(lines[shortline], real_dif, min_jiajiao, new_angle)
            # newline[4] = all_dif[all_jj.index(min_jiajiao)][1]  # 新的角度
            lines[shortline] = newline

        # for line in lines:
        #     plt.plot([line[1][0], line[2][0]], [line[1][1], line[2][1]])
        # plt.show()

    #     剔除冗余平行线，或在错误地方加入短线
        nnode = 0
        nline = len(lines)  # 线段个数
        while nnode < nline-1:
            print(nnode, nline)
            if abs(lines[nnode][4] - lines[nnode+1][4]) > 1:
                # print(abs(lines[nnode][4] - lines[nnode+1][4]))
                nnode = nnode+1
            else:
                #   计算平行线的距离
                prall_dis = dis_parr(lines[nnode][1], lines[nnode][2], lines[nnode+1][1], lines[nnode+1][2])
                print(prall_dis)
                if prall_dis > 0.5:
                    add_line = add_vertical(lines[nnode][0], lines[nnode][2], lines[nnode][4])
                    lines.insert(lines[nnode][0]+1, add_line)
                    lines = Sort_number(lines)
                    nnode = nnode + 1
                    nline = nline + 1
                else:
                    li = length(lines[nnode][1][0], lines[nnode][1][1], lines[nnode][2][0], lines[nnode][2][1])
                    li2 = length(lines[nnode+1][1][0], lines[nnode+1][1][1], lines[nnode+1][2][0], lines[nnode+1][2][1])
                    if li >= li2:
                        lines.remove(lines[nnode+1])
                    else:
                        lines.remove(lines[nnode])
                    lines = Sort_number(lines)
                    nline = nline - 1

        # 判断第一条边和最后一条边
        if abs(lines[-1][4] - lines[0][4]) < 1:
            prall_dis = dis_parr(lines[-1][1], lines[-1][2], lines[0][1], lines[0][2])
            if prall_dis > 0.5:
                add_line = add_vertical(lines[-1][0], lines[-1][2], lines[-1][4])
                lines.insert(lines[-1][0] + 1, add_line)
                lines = Sort_number(lines)
            else:
                li = length(lines[-1][1][0], lines[-1][1][1], lines[-1][2][0], lines[-1][2][1])
                li2 = length(lines[0][1][0], lines[0][1][1], lines[0][2][0], lines[0][2][1])
                if li >= li2:
                    lines.remove(lines[0])
                else:
                    lines.remove(lines[-1])
                lines = Sort_number(lines)
        # for line in lines:
        #     plt.plot([line[1][0], line[2][0]], [line[1][1], line[2][1]])
        # plt.show()


        print(len(lines))
        if len(lines) < 3:
            continue

        nodes = []
        for i in range(len(lines)-1):
            x, y = GetIntersectPointofLines(lines[i][1], lines[i][2], lines[i + 1][1], lines[i + 1][2])
            nodes.append((x, y))
        nodes.append(GetIntersectPointofLines(lines[0][1], lines[0][2], lines[len(lines)-1][1], lines[len(lines)-1][2]))

        # for line in lines:
        #     plt.plot([line[1][0], line[2][0]], [line[1][1], line[2][1]])
        # for i in range(len(nodes)-1):
        #     plt.plot([nodes[i][0], nodes[i+1][0]], [nodes[i][1], nodes[i+1][1]])
        # plt.plot([nodes[0][0], nodes[len(nodes)-1][0]], [nodes[0][1], nodes[len(nodes)-1][1]])
        # plt.show()

        w.poly(parts=[nodes])
        w.record(len(nodes))
    w.save('F:\\regularize\\21.tif')
'''


if __name__ == '__main__':
    group()

