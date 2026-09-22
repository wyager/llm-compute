; sum = 1 + 2 + ... + n
let n   = 1
let sum = 2
loop:
  jz  n, done
  add sum, sum, n
  sub n, n, #1
  jmp loop
done:
  halt
