"""
@File    :   iou.py
@Contact :   pengtt0119@gmail.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/27 2:54 下午   ttpeng      1.0         None
"""

def computing_iou(rec1, rec2):
	"""
	computing iou of two rectangle
	:param rec1: (x0, y0, x1, y1), which means(left, top, right, bottom)
	:param rec2: (x0, y0, x1, y1)
	"""

	# 1. computing area of each rectangle
	area_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
	area_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec1[1])

	# 2. computing sum of these two rectangles
	area_sum = area_rec1 + area_rec2

	# 3. find four edges of intersect rectangle
	left = max(rec1[0], rec2[0])
	top = max(rec1[1], rec2[1])
	right = min(rec1[2], rec2[2])
	bottom = min(rec1[3], rec2[3])

	# 4. determining if there is an intersection
	if top >= bottom or left >= right:
		return 0
	else:
		intersect = (bottom - top) * (right - left)

	# 5. computing the result of iou
	iou = intersect / (area_sum - intersect) * 1.0
	return iou


if __name__ == '__main__':

	rect1 = (361, 30, 450, 48)
	rect2 = (362, 30, 455, 48)
	iou = computing_iou(rect1, rect2)

	print(iou)

