# utilities for patch-related image processing

def image_vector_converter_pair(image_size):
  num_elems = reduce( lambda x,y: x*y, image_size )
  image_to_vector = lambda x: x.reshape((num_elems,))
  vector_to_image = lambda x: x.reshape(image_size)
  return image_to_vector, vector_to_image

def extract_patch(image, patch):
  return image[ patch[0][0]:patch[1][0], \
      patch[0][1]:patch[1][1] ]

def insert_patch(image, coords, patch):
  image[ coords[0][0]:coords[1][0], \
      coords[0][1]:coords[1][1] ] = patch

def patch_generator(shape, r=2, d=-1):
  """Generate patch boundary coordinates.

  Returned patch boundaries are tuples of the form
  ( (x0, y0), (x1, y1) ), and the patch can be extracted via the slice
  I[x0:x1, y0:y1].
  
  shape -- shape of space being partitioned, (Height, Width)
  r -- radius of partition.  For example, r=2 corresponds to 5x5
    patches; defaults to 5x5 patches
  d -- stride between patches.  For strictly non-overlapping patches,
    set this to 2*r + 1; this is the default

  """
  if d == -1: d = 2*r + 1

  (H, W) = shape
  for x in range(r, W-r, d):
    for y in range(r, H-r, d):
      yield ((x-r,y-r), (x+r+1,y+r+1))

# TODO
# write these to convert to/from matrices, then make sure that ksvd can
# represent an image well at all!!
def image_to_vectors(image, patch_generator):
  image_to_vector, vector_to_image = \
      image_vector_converter_pair( image.shape )
  return ( image_to_vector(extract_patch(image, pcoords)) \
      for pcoords in patch_generator )

def vectors_to_image(patches, shape, patch_generator):
  import numpy
  to_return = numpy.zeros(shape)
  for (vec, pcoords) in zip( 

