import shapefile, math, glob
import numpy as np
# # 读取shp文件
# shp = shapefile.Reader("F:/shp/areapart.shp")
# shapes = shp.shapes()
# # 新建存储文件
# w = shapefile.Writer(shapefile.POLYGON)
# w.autoBalance = 1
# w.field('FIRST_FLD', 'C', '40')
#
# for i in range(len(shapes)):
#     old = shapes[i].points[:-1]
#     num = len(old)
#     n = 0
#     while n < num:
#         bx = old[n][0]
#         by = old[n][1]
#         if n == 0:
#             ax = old[-1][0]
#             ay = old[-1][1]
#         else:
#             ax = old[n-1][0]
#             ay = old[n-1][1]
#         if n == len(old)-1:
#             cx = old[0][0]
#             cy = old[0][1]
#         else:
#             cx = old[n+1][0]
#             cy = old[n+1][1]
#         a2 = np.square(cx-bx) + np.square(cy-by)
#         a = np.sqrt(a2)
#         c2 = np.square(bx-ax) + np.square(by-ay)
#         c = np.sqrt(c2)
#         b2 = np.square(cx-ax) + np.square(cy-ay)
#         cosB = (a2+c2-b2)/(2*a*c)
#         if cosB > 1:
#             cosB = 1
#         if cosB < -1:
#             cosB = -1
#         B = math.degrees(math.acos(cosB))
#         if B < 40 or B > 170:
#             del old[n]
#             num = num - 1
#             if n > 0:
#                 n = n - 1
#         else:
#             n = n + 1
#
#     old.append(old[0])
#     if len(old) > 2:
#         w.poly(parts=[old])
#         w.record('')
# w.save("F:\\shp\\dg3P2.shp")


def com_area(contour):
    n = len(contour)
    s = 0
    for i in range(n-1):
        s = s + contour[i][0]*contour[i+1][1]-contour[i+1][0]*contour[i][1]
    s = s + contour[n-1][0]*contour[0][1]-contour[0][0]*contour[n-1][1]
    s = math.fabs(s)/2
    return s


def com_perimeter(contour):
    n = len(contour)
    l = 0
    for i in range(n-1):
        l = l + np.sqrt(np.square(contour[i+1][0]-contour[i][0]) + np.square(contour[i+1][1]-contour[i][1]))
    l = l + np.sqrt(np.square(contour[n-1][0]-contour[0][0]) + np.square(contour[n-1][1]-contour[0][1]))
    return l

#  根据不同面积，剔除太短的边，剔除过于尖锐的角和过于的平角
def shpsimple():
    shps = glob.glob("F:\\test\\test1_full2.shp")
    for shp in shps:
        s = shapefile.Reader(shp)
        shapes = s.shapes()

        w = shapefile.Writer(shapefile.POLYGON)
        w.autoBalance = 1
        w.field('FIRST_FLD', 'C', '40')

        for i in range(len(shapes)-1):
            initial = shapes[i].points[:-1]
            area = com_area(initial)
            after = []
            # if area <= 10:
            #     continue
            # elif 10 < area <= 500:
            #     after.append(initial[0])
            #     for pn in range(1, len(initial)):
            #         length = np.sqrt(np.square(initial[pn][0]-initial[pn-1][0]) + np.square(initial[pn][1]-initial[pn-1][1]))
            #         if length > 0.5:
            #             after.append(initial[pn])
            # elif 500 < area <= 1000:
            #     after.append(initial[0])
            #     for pn in range(1, len(initial)):
            #         length = np.sqrt(np.square(initial[pn][0]-initial[pn-1][0]) + np.square(initial[pn][1]-initial[pn-1][1]))
            #         if length > 0.75:
            #             after.append(initial[pn])
            # elif 1000 < area <= 5000:
            #     after.append(initial[0])
            #     for pn in range(1, len(initial)):
            #         length = np.sqrt(np.square(initial[pn][0]-initial[pn-1][0]) + np.square(initial[pn][1]-initial[pn-1][1]))
            #         if length > 1:
            #             after.append(initial[pn])
            # elif 5000 < area <= 10000:
            #     after.append(initial[0])
            #     for pn in range(1, len(initial)):
            #         length = np.sqrt(np.square(initial[pn][0]-initial[pn-1][0]) + np.square(initial[pn][1]-initial[pn-1][1]))
            #         if length > 1.25:
            #             after.append(initial[pn])
            # elif area > 10000:
            #     after.append(initial[0])
            #     for pn in range(1, len(initial)):
            #         length = np.sqrt(np.square(initial[pn][0]-initial[pn-1][0]) + np.square(initial[pn][1]-initial[pn-1][1]))
            #         if length > 1.5:
            #             after.append(initial[pn])
            if area <= 10:
                continue
            elif area > 10:
                after.append(initial[0])
                for pn in range(1, len(initial)):
                    length = np.sqrt(np.square(initial[pn][0]-initial[pn-1][0]) + np.square(initial[pn][1]-initial[pn-1][1]))
                    if length > 0.5:
                        after.append(initial[pn])
            old = after
            num = len(old)
            n = 0
            if num < 100000:
                while n < num:
                    bx = old[n][0]
                    by = old[n][1]
                    if n == 0:
                        ax = old[-1][0]
                        ay = old[-1][1]
                    else:
                        ax = old[n - 1][0]
                        ay = old[n - 1][1]
                    if n == len(old) - 1:
                        cx = old[0][0]
                        cy = old[0][1]
                    else:
                        cx = old[n + 1][0]
                        cy = old[n + 1][1]
                    a2 = np.square(cx - bx) + np.square(cy - by)
                    a = np.sqrt(a2)
                    c2 = np.square(bx - ax) + np.square(by - ay)
                    c = np.sqrt(c2)
                    b2 = np.square(cx - ax) + np.square(cy - ay)
                    cosB = (a2 + c2 - b2) / (2 * a * c)
                    if cosB > 1:
                        cosB = 1
                    if cosB < -1:
                        cosB = -1
                    B = math.degrees(math.acos(cosB))
                    if B < 40 or B > 170:
                        del old[n]
                        num = num - 1
                        if n > 0:
                            n = n - 1
                    else:
                        n = n + 1
                old.append(old[0])
                if len(old) > 2:
                    w.poly(parts=[old])
                    w.record(len(old)-1)
        w.save(shp[:-4]+"1.shp")
        print(shp)
        return w

# 根据面积和周长与点数的比值，剔除错误的提取
def ratio(shp):
    # s = shapefile.Reader(shp)
    shapes = shp.shapes()

    w = shapefile.Writer(shapefile.POLYGON)
    w.autoBalance = 1
    w.field('FIRST_FLD', 'C', '40')
    for i in range(len(shapes) - 1):
        initial = shapes[i].points[:-1]
        pn = len(initial)
        area = com_area(initial)
        perimeter = com_perimeter(initial)
        ap = area/pn
        pp = perimeter/pn

        if area <= 20:
            continue
        else:
            if ap > 1 or pp > 1.5:
                w.poly(parts=[shapes[i].points])
                w.record('')
    w.save("F:\\test\\last1.shp")


if __name__ == '__main__':
    shp = shpsimple()
    ratio(shp)