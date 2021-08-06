function fixname(s,n,t) {
  n = match(s, / *$/);
  if (n == 1) n = 2;
  t=substr(s, 1, n-1);
  gsub(/ /, "_", t);
  return t substr(s, n);
}
function fixfield(s,a,b) {
  if (length(s) <= a || match(substr(s, a), " ") == 0) {
    return s;
  }
  return substr(s, 1, a-1) fixname(substr(s, a, b-a+1)) substr(s, b+1);
}
{
  if (substr($0, 1, 1) != " ") {
    print $0;
  } else {
    s = fixfield($0, 5, 12);
    s = fixfield(s, 15, 22);
    s = fixfield(s, 40, 47);
    print s;
  }
}
