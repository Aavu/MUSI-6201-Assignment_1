# MUSI-6201-Assignment_1

## Modification of Pitch Tracker:
### Interpolation to ACF curve
To increase the precision of the identified frame-based pitch, apply parabolic interpolation around the maximum of the ACF curve. In particular, if n0 is the maximum index of an ACF curve, then we can fit a parabola that passes three points (n0−1,ACF(n0−1)), (n0,ACF(n0)), and (n0+1,ACF(n0+1)), and then use the maxmizing position of this parabola to compute the pitch.<br>
Theorectically speaking, I think this approach is correct, but I doubt whether it is useful practically.

### Medium filter
To smooth the pitch curve such that abrupt-changing pitch are removed. Set the kernel size to 3 or 5.