/* The Image Registration Toolkit (IRTK)
 *
 * Copyright 2008-2015 Imperial College London
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#ifndef _IRTKLINEARIMAGEGRADIENTFUNCTION_HXX
#define _IRTKLINEARIMAGEGRADIENTFUNCTION_HXX

#include <irtkLinearImageGradientFunction.h>
#include <irtkImageGradientFunction.hxx>
#include <irtkLinearInterpolateImageFunction.hxx>


// =============================================================================
// Construction/Destruction
// =============================================================================

// -----------------------------------------------------------------------------
template <class TImage>
irtkGenericLinearImageGradientFunction<TImage>
::irtkGenericLinearImageGradientFunction()
{
}

// -----------------------------------------------------------------------------
template <class TImage>
irtkGenericLinearImageGradientFunction<TImage>
::~irtkGenericLinearImageGradientFunction()
{
}

// -----------------------------------------------------------------------------
template <class TImage>
void irtkGenericLinearImageGradientFunction<TImage>::Initialize(bool coeff)
{
  /// Initialize base class
  Superclass::Initialize(coeff);

  // Domain on which linear interpolation is defined: [.5, x-.5)
  switch (this->NumberOfDimensions()) {
    case 3:
      this->_z1 = .5;
      this->_z2 = fdec(this->Input()->Z() - 1.5);
    default:
      this->_y1 = .5;
      this->_y2 = fdec(this->Input()->Y() - 1.5);
      this->_x1 = .5;
      this->_x2 = fdec(this->Input()->X() - 1.5);
  }

  // Initialize interpolator
  _ContinuousImage.Input(this->Input());
  _ContinuousImage.Initialize(coeff);
}

// =============================================================================
// Domain checks
// =============================================================================

// -----------------------------------------------------------------------------
template <class TImage>
void irtkGenericLinearImageGradientFunction<TImage>
::BoundingInterval(double x, int &i, int &I) const
{
  i = static_cast<int>(floor(x - .5)), I = i + 2;
}

// =============================================================================
// Evaluation
// =============================================================================

// -----------------------------------------------------------------------------
template <class TImage>
inline typename irtkGenericLinearImageGradientFunction<TImage>::GradientType
irtkGenericLinearImageGradientFunction<TImage>
::GetInside2D(double x, double y, double z, double t) const
{
  GradientType val;
  val._x = _ContinuousImage.GetInside(x + .5, y, z, t) -
           _ContinuousImage.GetInside(x - .5, y, z, t);
  val._y = _ContinuousImage.GetInside(x, y + .5, z, t) -
           _ContinuousImage.GetInside(x, y - .5, z, t);
  return val;
}

// -----------------------------------------------------------------------------
template <class TImage>
inline typename irtkGenericLinearImageGradientFunction<TImage>::GradientType
irtkGenericLinearImageGradientFunction<TImage>
::GetWithPaddingInside2D(double x, double y, double z, double t) const
{
  GradientType val;
  VoxelType a, b;

  a = _ContinuousImage.GetWithPaddingInside(x - .5, y, z, t);
  if (a == _ContinuousImage.DefaultValue()) return this->DefaultValue();
  b = _ContinuousImage.GetWithPaddingInside(x + .5, y, z, t);
  if (b == _ContinuousImage.DefaultValue()) return this->DefaultValue();
  val._x = b - a;

  a = _ContinuousImage.GetWithPaddingInside(x, y - .5, z, t);
  if (a == _ContinuousImage.DefaultValue()) return this->DefaultValue();
  b = _ContinuousImage.GetWithPaddingInside(x, y + .5, z, t);
  if (b == _ContinuousImage.DefaultValue()) return this->DefaultValue();
  val._y = b - a;

  return val;
}

// -----------------------------------------------------------------------------
template <class TImage>
inline typename irtkGenericLinearImageGradientFunction<TImage>::GradientType
irtkGenericLinearImageGradientFunction<TImage>
::GetInside3D(double x, double y, double z, double t) const
{
  GradientType val;
  val._x = _ContinuousImage.GetInside(x + .5, y, z, t) -
           _ContinuousImage.GetInside(x - .5, y, z, t);
  val._y = _ContinuousImage.GetInside(x, y + .5, z, t) -
           _ContinuousImage.GetInside(x, y - .5, z, t);
  val._z = _ContinuousImage.GetInside(x, y, z + .5, t) -
           _ContinuousImage.GetInside(x, y, z - .5, t);
  return val;
}

// -----------------------------------------------------------------------------
template <class TImage>
inline typename irtkGenericLinearImageGradientFunction<TImage>::GradientType
irtkGenericLinearImageGradientFunction<TImage>
::GetWithPaddingInside3D(double x, double y, double z, double t) const
{
  GradientType val;
  VoxelType a, b;

  a = _ContinuousImage.GetWithPaddingInside(x - .5, y, z, t);
  if (a == _ContinuousImage.DefaultValue()) return this->DefaultValue();
  b = _ContinuousImage.GetWithPaddingInside(x + .5, y, z, t);
  if (b == _ContinuousImage.DefaultValue()) return this->DefaultValue();
  val._x = b - a;

  a = _ContinuousImage.GetWithPaddingInside(x, y - .5, z, t);
  if (a == _ContinuousImage.DefaultValue()) return this->DefaultValue();
  b = _ContinuousImage.GetWithPaddingInside(x, y + .5, z, t);
  if (b == _ContinuousImage.DefaultValue()) return this->DefaultValue();
  val._y = b - a;

  a = _ContinuousImage.GetWithPaddingInside(x, y, z - .5, t);
  if (a == _ContinuousImage.DefaultValue()) return this->DefaultValue();
  b = _ContinuousImage.GetWithPaddingInside(x, y, z + .5, t);
  if (b == _ContinuousImage.DefaultValue()) return this->DefaultValue();
  val._z = b - a;

  return val;
}

// -----------------------------------------------------------------------------
template <class TImage>
inline typename irtkGenericLinearImageGradientFunction<TImage>::GradientType
irtkGenericLinearImageGradientFunction<TImage>
::GetInside(double x, double y, double z, double t) const
{
  switch (this->NumberOfDimensions()) {
    case 2:  return GetInside2D(x, y, z, t);
    default: return GetInside3D(x, y, z, t);
  }
}

// -----------------------------------------------------------------------------
template <class TImage>
inline typename irtkGenericLinearImageGradientFunction<TImage>::GradientType
irtkGenericLinearImageGradientFunction<TImage>
::GetOutside(double x, double y, double z, double t) const
{
  return this->DefaultValue();
}

// -----------------------------------------------------------------------------
template <class TImage>
inline typename irtkGenericLinearImageGradientFunction<TImage>::GradientType
irtkGenericLinearImageGradientFunction<TImage>
::GetWithPaddingInside(double x, double y, double z, double t) const
{
  switch (this->NumberOfDimensions()) {
    case 2:  return GetWithPaddingInside2D(x, y, z, t);
    default: return GetWithPaddingInside3D(x, y, z, t);
  }
}

// -----------------------------------------------------------------------------
template <class TImage>
inline typename irtkGenericLinearImageGradientFunction<TImage>::GradientType
irtkGenericLinearImageGradientFunction<TImage>
::GetWithPaddingOutside(double x, double y, double z, double t) const
{
  return this->DefaultValue();
}


#endif
