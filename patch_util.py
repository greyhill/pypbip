# utilities for patch-related image processing

def image_vector_converter_pair(image_size):
  num_elems = reduce( lambda x,y: x*y, image_size )
  image_to_vector = lambda x: x.reshape((num_elems,1))
  vector_to_image = lambda x: x.reshape(image_size)
  return image_to_vector, vector_to_image

def extract_patch(image, patch):
  return image[ patch[0][0]:patch[1][0], \
      patch[0][1]:patch[1][1] ]

def insert_patch(image, coords, patch):
  image[ coords[0][0]:coords[1][0], \
      coords[0][1]:coords[1][1] ] = patch

class patch_generator(object):
  def __init__(self, shape, r=2, d=-1):
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
    self.radius = r
    if d == -1:
      self.stride = 2*r+1
    else:
      self.stride = d
    self.image_shape = shape
    self.patch_shape = ( r*2+1, 2*r+1 )

  def __iter__(self):
    (H, W) = self.image_shape
    for x in range(self.radius, W-self.radius, self.stride):
      for y in range(self.radius, H-self.radius, self.stride):
        yield ((x-self.radius,y-self.radius), \
            (x+self.radius+1,y+self.radius+1))

def column_seq(matrix):
  """Generates a sequence from the columns of the given matrix."""
  return ( matrix[:, i] for i in range(matrix.shape[1]) )

def image_to_vectors(image, patch_generator):
  """Returns a sequence of vectors forming the patches of the image as
  specified by the given patch_generator."""
  image_to_vector, vector_to_image = \
      image_vector_converter_pair( patch_generator.patch_shape )
  return ( image_to_vector(extract_patch(image, pcoords)) \
      for pcoords in patch_generator )

def disjoint_vectors_to_image(patches, patch_generator):
  """Combines a sequence of disjoint patch vectors into an image."""
  import numpy
  to_return = numpy.zeros(patch_generator.image_shape)
  image_to_vector, vector_to_image = \
      image_vector_converter_pair( patch_generator.patch_shape )
  for (vec, pcoords) in zip(patches, patch_generator):
    insert_patch(to_return, pcoords, vector_to_image(vec))
  return to_return

