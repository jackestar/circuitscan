# Format Maker

Given the connection format generated by the model, the ```formatmaker``` program will generate a file for the circuit simulator

## Input format
```
always (mandatory) [optional]

Node Declaration
N(nodenumber)

nodenumber        node number (1 to 99)

Component
(type)(Number) (ConnectionNode) (ConnectionNode) [PosX] [PosY] [Orientation]

type              component type
    R to resistance
    L to inductor
    C to capacitor
    V to voltage source
number            number of compoenent
ConnectionNode    node to which is connected
PosX              X position in the plane (optional)
PosY              Y position in the plane (optional)
Orientation       Orientation of the component (0 to 3) (optional)

// Future considerations
Node/Componente Name
N(nodenumber).name string
(type)(Number).name string


string            string without spaces with the node name

Componente Propities
(type)(Number).value value

value             a float value (exponentials can be valid)
```

The algorithm may not give a value for any of the [optional] fields, so the program must be able to assign some value, whether random or calculated.

Example

12v Voltage source, 23Ohm resistance and 20mH inductor in series
```
N1
N2
N3

V1 N1 N3
R1 N1 N2
L1 N2 N3

V1.value 12
R1.value 23
L1.value 2e-2
```

## Output format

these are the candidates for output formats\
[ngspice](https://ngspice.sourceforge.io/)\
[circuitjs](https://github.com/pfalstad/circuitjs1)\
\
On the one hand, ngspice has a widely used and documented format, the only problem will be the complexity of the simulator and its implementation with the finished application, on the other hand, circuitjs is easy to implement in a web environment with the only difficulty being that it does not have much documentation.