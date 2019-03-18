Gamma spots removal for Neutron radiography and tomography images
using 'find and replace' strategy, this is a discriminative method,
better than unique threshold substitution

Find the gamma spots by laplacian or LOG operation, which are usually utilized for
edge finding. Those greatly changing area will produce high level in the resultant image. This can work very well for those high level gamma spots.
For those relatively low level gamma spots, a lower threshold is needed. But,
some steep edges also have very high values. This make it a hard choice: whether filter out those
low level noise at some sacrifice of the edge info, or keep the edge intact as well as those low level noise.

adptive thresholding : med3(log)+ thr
based on the fact that: the edges of the obj usually have a width of more than 5 pixels,
while the gamma spots usually involve less then 3 pixels in width. By filtering the
resultant image from LOG(laplacian of gaussian) filtering with a 3by3 median kernel, the edges
will survive, while most of the gamma pixels lost their magnitude drastically, some are wiped out.
Adaptive size of median filter:
   3 by 3 for small spots, 5 by 5 for medium ones, 7 by 7 for high level ones.

written by Hongyun Li, visiting physicist at FRM2,TUM,Germany, Feb 09, 2006
Contact info:
hongyunlee@yahoo.com, or lihongyun03@mails.tsinghua.edu.cn
Northwest Institute of Nuclear Technology, China

ported to cuda by Yawar Siddiqui    