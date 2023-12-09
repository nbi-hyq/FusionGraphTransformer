import shapely
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd


def circle(x=0.0, y=0.0, r=1.0, tol=0.001):
	p = shapely.Point(x, y)
	x = p.buffer(r)
	return x.simplify(tol, preserve_topology=False)

def line(l_pt, r=1.0):
	line = shapely.LineString(l_pt)
	return line.buffer(r, single_sided=False)

def interset_all(l_poly, ax=None):
	l_inter = []
	for i in range(len(l_poly)):
		if ax:
			poly = Polygon(shapely.get_coordinates(l_poly[i]), fill=None)
			ax.add_patch(poly)
		for j in range(i+1, len(l_poly)):
			if l_poly[i].intersects(l_poly[j]):
				inter = l_poly[i].intersection(l_poly[j])
				l_inter.append(inter)
				if ax:
					poly = Polygon(shapely.get_coordinates(inter), fill=None)
					ax.add_patch(poly)
	return l_inter

def plot_all_poly(l_poly, ax):
	for i in range(len(l_poly)):
		poly = Polygon(shapely.get_coordinates(l_poly[i]), fill=None)
		ax.add_patch(poly)

if __name__ == '__main__':
	fig, ax1 = plt.subplots(1,1)

	c0 = circle(x=-2, r=3)
	c1 = circle(x=2, r=3)
	c2 = circle(y=-2, r=1.5)
	l1 = line([(-3.5, 0), (0.7, 1.4), (-3.5, 0), (0, 4)], r=0.3)
	l2 = line([(3.5, 0), (-0.7, 1.3), (3.5, 0), (0.2, 4)], r=0.5)
	poly_list = [c0, c1, c2, l1, l2]

	diff_mod2 = shapely.symmetric_difference_all(poly_list)  # compute/plot intersection of intersections
	myPoly = gpd.GeoSeries([diff_mod2])
	myPoly.plot(ax=ax1)
	plot_all_poly(poly_list, ax1)
	
	plt.ylim(-10,10)
	plt.xlim(-10,10)
	plt.savefig('test.pdf')
	#plt.show()

