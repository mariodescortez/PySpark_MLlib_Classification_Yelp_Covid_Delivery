# PySpark_MLlib_Classification_Yelp_Covid Delivery

ML to predict whether a business  will start doing delivery or takeout after a lockdown - PySpark

# Requirements

  * Python 3.7

> This code makes use of the `f"..."` or [f-string
> syntax](https://www.python.org/dev/peps/pep-0498/). This syntax was
> introduced in Python 3.6.


# Sample Execution & Output

If run without command line arguments, using

```
./precisionEstimate
```

the following usage message will be displayed.

```
Usage: ./estimatePrecision.py num_execs [arbitrary precision]
```

If run using

```
./precisionEstimate 1000000
```

output *simliar* to

```
           float|  3.7247|2.220446049250313e-16
 float-type-hint|  3.6731|2.220446049250313e-16
      Decimal-28| 31.8602|0.999999999999999999999999999
```

will be generated. Note that the `float` and `float-type-hint` lines may vary.

---

An optional precision command line argument can be supplied to change the
arbitrary precision used by the Python `decimal` module. For example:

```
./precisionEstimate 1000000 16
```

will generate output similar to

```
           float| 0.3979|2.220446049250313e-16
 float-type-hint| 0.4053|2.220446049250313e-16
      Decimal-16| 3.1643|0.999999999999999
```

