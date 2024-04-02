[define indoor_room]
indoor_room is a kind of room.
indoor_room has a text called description.
indoor_room has a text called printed name.

[define area]
area is a kind of room.
area has a text called description.
area has a text called printed name.
area has a indoor_room called parent.
area has a text called X.
area has a text called Y.
area can be enterable.
area is always enterable.

player has a number called orientation.


[define landmark]
landmark is a kind of thing.
landmark has a text called description.
landmark has a text called printed name.
landmark can be examined.
area has a list of landmark called visible_objects.

[create r0]
r0 is a indoor_room. "Room 0".

[create r1]
r1 is a indoor_room. "Room 1".

[create r2]
r2 is a indoor_room. "Room 2".

[create r3]
r3 is a indoor_room. "Room 3".

[create r4]
r4 is a indoor_room. "Room 4".

[create r5]
r5 is a indoor_room. "Room 5".

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "547105.9412801263".
the Y of the a4r0 is "1908354.6105585739".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "547187.2996006935".
the Y of the a5r0 is "1908518.356759838".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "545497.3114406405".
the Y of the a8r0 is "1908578.958403625".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "546881.5201489937".
the Y of the a11r0 is "1908614.7111584658".

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "546915.7072982964".
the Y of the a12r0 is "1908792.236317469".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "547185.875".
the Y of the a17r0 is "1908250.0994746273".

[create a18r0]
a18r0 is a area. "An area (18) in r0".
the printed name of the a18r0 is "Room 0".
Understand "Area 18 in Room 0" as a18r0.
the parent of the a18r0 is r0.
the X of the a18r0 is "545871.6022888338".
the Y of the a18r0 is "1908342.1318872178".

[create a21r0]
a21r0 is a area. "An area (21) in r0".
the printed name of the a21r0 is "Room 0".
Understand "Area 21 in Room 0" as a21r0.
the parent of the a21r0 is r0.
the X of the a21r0 is "546483.9818580335".
the Y of the a21r0 is "1908338.5786448321".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "547046.9184555504".
the Y of the a22r0 is "1908353.5565795635".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "546963.125".
the Y of the a23r0 is "1908242.750504866".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "547060.4986470155".
the Y of the a44r0 is "1908866.5019235234".

[create a60r0]
a60r0 is a area. "An area (60) in r0".
the printed name of the a60r0 is "Room 0".
Understand "Area 60 in Room 0" as a60r0.
the parent of the a60r0 is r0.
the X of the a60r0 is "545831.0".
the Y of the a60r0 is "1908200.0".

[create a61r0]
a61r0 is a area. "An area (61) in r0".
the printed name of the a61r0 is "Room 0".
Understand "Area 61 in Room 0" as a61r0.
the parent of the a61r0 is r0.
the X of the a61r0 is "545866.6000000001".
the Y of the a61r0 is "1908325.5734335838".

[create a64r0]
a64r0 is a area. "An area (64) in r0".
the printed name of the a64r0 is "Room 0".
Understand "Area 64 in Room 0" as a64r0.
the parent of the a64r0 is r0.
the X of the a64r0 is "547069.0".
the Y of the a64r0 is "1908508.0".

[create a65r0]
a65r0 is a area. "An area (65) in r0".
the printed name of the a65r0 is "Room 0".
Understand "Area 65 in Room 0" as a65r0.
the parent of the a65r0 is r0.
the X of the a65r0 is "547089.624563081".
the Y of the a65r0 is "1908541.1476019477".

[create a68r0]
a68r0 is a area. "An area (68) in r0".
the printed name of the a68r0 is "Room 0".
Understand "Area 68 in Room 0" as a68r0.
the parent of the a68r0 is r0.
the X of the a68r0 is "545386.0".
the Y of the a68r0 is "1908485.0".

[create a69r0]
a69r0 is a area. "An area (69) in r0".
the printed name of the a69r0 is "Room 0".
Understand "Area 69 in Room 0" as a69r0.
the parent of the a69r0 is r0.
the X of the a69r0 is "545483.3466845371".
the Y of the a69r0 is "1908567.0099861317".

[create a75r0]
a75r0 is a area. "An area (75) in r0".
the printed name of the a75r0 is "Room 0".
Understand "Area 75 in Room 0" as a75r0.
the parent of the a75r0 is r0.
the X of the a75r0 is "546530.0".
the Y of the a75r0 is "1908200.0".

[create a76r0]
a76r0 is a area. "An area (76) in r0".
the printed name of the a76r0 is "Room 0".
Understand "Area 76 in Room 0" as a76r0.
the parent of the a76r0 is r0.
the X of the a76r0 is "546495.7".
the Y of the a76r0 is "1908303.4552610856".

[create a87r0]
a87r0 is a area. "An area (87) in r0".
the printed name of the a87r0 is "Room 0".
Understand "Area 87 in Room 0" as a87r0.
the parent of the a87r0 is r0.
the X of the a87r0 is "546926.0".
the Y of the a87r0 is "1908200.0".

[create a90r0]
a90r0 is a area. "An area (90) in r0".
the printed name of the a90r0 is "Room 0".
Understand "Area 90 in Room 0" as a90r0.
the parent of the a90r0 is r0.
the X of the a90r0 is "547223.0".
the Y of the a90r0 is "1908200.0".

[create a91r0]
a91r0 is a area. "An area (91) in r0".
the printed name of the a91r0 is "Room 0".
Understand "Area 91 in Room 0" as a91r0.
the parent of the a91r0 is r0.
the X of the a91r0 is "547097.0".
the Y of the a91r0 is "1908895.0".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "547828.8305982074".
the Y of the a0r1 is "1910298.8947038355".

[create a6r1]
a6r1 is a area. "An area (6) in r1".
the printed name of the a6r1 is "Room 1".
Understand "Area 6 in Room 1" as a6r1.
the parent of the a6r1 is r1.
the X of the a6r1 is "547692.5565103933".
the Y of the a6r1 is "1910302.537885611".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "547901.9000000004".
the Y of the a34r1 is "1910389.4667546176".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "547621.1000000002".
the Y of the a35r1 is "1910389.5239782464".

[create a84r1]
a84r1 is a area. "An area (84) in r1".
the printed name of the a84r1 is "Room 1".
Understand "Area 84 in Room 1" as a84r1.
the parent of the a84r1 is r1.
the X of the a84r1 is "547937.0".
the Y of the a84r1 is "1910440.0".

[create a85r1]
a85r1 is a area. "An area (85) in r1".
the printed name of the a85r1 is "Room 1".
Understand "Area 85 in Room 1" as a85r1.
the parent of the a85r1 is r1.
the X of the a85r1 is "547586.0".
the Y of the a85r1 is "1910440.0".

[create a1r2]
a1r2 is a area. "An area (1) in r2".
the printed name of the a1r2 is "Room 2".
Understand "Area 1 in Room 2" as a1r2.
the parent of the a1r2 is r2.
the X of the a1r2 is "548402.2576810806".
the Y of the a1r2 is "1908353.704645209".

[create a2r2]
a2r2 is a area. "An area (2) in r2".
the printed name of the a2r2 is "Room 2".
Understand "Area 2 in Room 2" as a2r2.
the parent of the a2r2 is r2.
the X of the a2r2 is "548450.9255680318".
the Y of the a2r2 is "1908354.3366956885".

[create a15r2]
a15r2 is a area. "An area (15) in r2".
the printed name of the a15r2 is "Room 2".
Understand "Area 15 in Room 2" as a15r2.
the parent of the a15r2 is r2.
the X of the a15r2 is "548532.0".
the Y of the a15r2 is "1908249.2614346421".

[create a16r2]
a16r2 is a area. "An area (16) in r2".
the printed name of the a16r2 is "Room 2".
Understand "Area 16 in Room 2" as a16r2.
the parent of the a16r2 is r2.
the X of the a16r2 is "548322.0".
the Y of the a16r2 is "1908249.422623434".

[create a88r2]
a88r2 is a area. "An area (88) in r2".
the printed name of the a88r2 is "Room 2".
Understand "Area 88 in Room 2" as a88r2.
the parent of the a88r2 is r2.
the X of the a88r2 is "548567.0".
the Y of the a88r2 is "1908200.0".

[create a89r2]
a89r2 is a area. "An area (89) in r2".
the printed name of the a89r2 is "Room 2".
Understand "Area 89 in Room 2" as a89r2.
the parent of the a89r2 is r2.
the X of the a89r2 is "548287.0".
the Y of the a89r2 is "1908200.0".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "548774.4341612108".
the Y of the a3r3 is "1909458.8649328328".

[create a9r3]
a9r3 is a area. "An area (9) in r3".
the printed name of the a9r3 is "Room 3".
Understand "Area 9 in Room 3" as a9r3.
the parent of the a9r3 is r3.
the X of the a9r3 is "549076.460768834".
the Y of the a9r3 is "1909436.6567164178".

[create a24r3]
a24r3 is a area. "An area (24) in r3".
the printed name of the a24r3 is "Room 3".
Understand "Area 24 in Room 3" as a24r3.
the parent of the a24r3 is r3.
the X of the a24r3 is "548914.4999999995".
the Y of the a24r3 is "1910391.1290622605".

[create a25r3]
a25r3 is a area. "An area (25) in r3".
the printed name of the a25r3 is "Room 3".
Understand "Area 25 in Room 3" as a25r3.
the parent of the a25r3 is r3.
the X of the a25r3 is "548988.4609628964".
the Y of the a25r3 is "1910301.4317349754".

[create a30r3]
a30r3 is a area. "An area (30) in r3".
the printed name of the a30r3 is "Room 3".
Understand "Area 30 in Room 3" as a30r3.
the parent of the a30r3 is r3.
the X of the a30r3 is "549297.1539155534".
the Y of the a30r3 is "1910075.8094368111".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "550250.5314170674".
the Y of the a31r3 is "1910303.189299077".

[create a33r3]
a33r3 is a area. "An area (33) in r3".
the printed name of the a33r3 is "Room 3".
Understand "Area 33 in Room 3" as a33r3.
the parent of the a33r3 is r3.
the X of the a33r3 is "549133.5".
the Y of the a33r3 is "1910412.9516454327".

[create a36r3]
a36r3 is a area. "An area (36) in r3".
the printed name of the a36r3 is "Room 3".
Understand "Area 36 in Room 3" as a36r3.
the parent of the a36r3 is r3.
the X of the a36r3 is "549070.4738833166".
the Y of the a36r3 is "1909650.8752076172".

[create a37r3]
a37r3 is a area. "An area (37) in r3".
the printed name of the a37r3 is "Room 3".
Understand "Area 37 in Room 3" as a37r3.
the parent of the a37r3 is r3.
the X of the a37r3 is "548885.7566097162".
the Y of the a37r3 is "1909889.65889394".

[create a38r3]
a38r3 is a area. "An area (38) in r3".
the printed name of the a38r3 is "Room 3".
Understand "Area 38 in Room 3" as a38r3.
the parent of the a38r3 is r3.
the X of the a38r3 is "548878.2774717974".
the Y of the a38r3 is "1909702.6137951824".

[create a39r3]
a39r3 is a area. "An area (39) in r3".
the printed name of the a39r3 is "Room 3".
Understand "Area 39 in Room 3" as a39r3.
the parent of the a39r3 is r3.
the X of the a39r3 is "550575.1472802768".
the Y of the a39r3 is "1909777.4773217994".

[create a40r3]
a40r3 is a area. "An area (40) in r3".
the printed name of the a40r3 is "Room 3".
Understand "Area 40 in Room 3" as a40r3.
the parent of the a40r3 is r3.
the X of the a40r3 is "550646.1924349966".
the Y of the a40r3 is "1909912.1997239743".

[create a42r3]
a42r3 is a area. "An area (42) in r3".
the printed name of the a42r3 is "Room 3".
Understand "Area 42 in Room 3" as a42r3.
the parent of the a42r3 is r3.
the X of the a42r3 is "550789.6090407216".
the Y of the a42r3 is "1909831.1038106412".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "550616.2356620994".
the Y of the a43r3 is "1910014.3319104235".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "548793.5386871184".
the Y of the a47r3 is "1909533.4011819737".

[create a50r3]
a50r3 is a area. "An area (50) in r3".
the printed name of the a50r3 is "Room 3".
Understand "Area 50 in Room 3" as a50r3.
the parent of the a50r3 is r3.
the X of the a50r3 is "550745.0".
the Y of the a50r3 is "1910106.0".

[create a51r3]
a51r3 is a area. "An area (51) in r3".
the printed name of the a51r3 is "Room 3".
Understand "Area 51 in Room 3" as a51r3.
the parent of the a51r3 is r3.
the X of the a51r3 is "550650.1699148411".
the Y of the a51r3 is "1910037.0028792769".

[create a56r3]
a56r3 is a area. "An area (56) in r3".
the printed name of the a56r3 is "Room 3".
Understand "Area 56 in Room 3" as a56r3.
the parent of the a56r3 is r3.
the X of the a56r3 is "549190.0".
the Y of the a56r3 is "1910178.0".

[create a57r3]
a57r3 is a area. "An area (57) in r3".
the printed name of the a57r3 is "Room 3".
Understand "Area 57 in Room 3" as a57r3.
the parent of the a57r3 is r3.
the X of the a57r3 is "549280.9423265149".
the Y of the a57r3 is "1910092.5787337879".

[create a58r3]
a58r3 is a area. "An area (58) in r3".
the printed name of the a58r3 is "Room 3".
Understand "Area 58 in Room 3" as a58r3.
the parent of the a58r3 is r3.
the X of the a58r3 is "550295.0".
the Y of the a58r3 is "1910440.0".

[create a59r3]
a59r3 is a area. "An area (59) in r3".
the printed name of the a59r3 is "Room 3".
Understand "Area 59 in Room 3" as a59r3.
the parent of the a59r3 is r3.
the X of the a59r3 is "550262.0".
the Y of the a59r3 is "1910336.770958084".

[create a72r3]
a72r3 is a area. "An area (72) in r3".
the printed name of the a72r3 is "Room 3".
Understand "Area 72 in Room 3" as a72r3.
the parent of the a72r3 is r3.
the X of the a72r3 is "548927.0".
the Y of the a72r3 is "1909336.0".

[create a73r3]
a73r3 is a area. "An area (73) in r3".
the printed name of the a73r3 is "Room 3".
Understand "Area 73 in Room 3" as a73r3.
the parent of the a73r3 is r3.
the X of the a73r3 is "548927.7705672854".
the Y of the a73r3 is "1909436.6567164178".

[create a74r3]
a74r3 is a area. "An area (74) in r3".
the printed name of the a74r3 is "Room 3".
Understand "Area 74 in Room 3" as a74r3.
the parent of the a74r3 is r3.
the X of the a74r3 is "548896.1468610526".
the Y of the a74r3 is "1909426.100308724".

[create a79r3]
a79r3 is a area. "An area (79) in r3".
the printed name of the a79r3 is "Room 3".
Understand "Area 79 in Room 3" as a79r3.
the parent of the a79r3 is r3.
the X of the a79r3 is "550547.0".
the Y of the a79r3 is "1909736.0".

[create a81r3]
a81r3 is a area. "An area (81) in r3".
the printed name of the a81r3 is "Room 3".
Understand "Area 81 in Room 3" as a81r3.
the parent of the a81r3 is r3.
the X of the a81r3 is "550834.0".
the Y of the a81r3 is "1909803.0".

[create a83r3]
a83r3 is a area. "An area (83) in r3".
the printed name of the a83r3 is "Room 3".
Understand "Area 83 in Room 3" as a83r3.
the parent of the a83r3 is r3.
the X of the a83r3 is "549170.0".
the Y of the a83r3 is "1910440.0".

[create a86r3]
a86r3 is a area. "An area (86) in r3".
the printed name of the a86r3 is "Room 3".
Understand "Area 86 in Room 3" as a86r3.
the parent of the a86r3 is r3.
the X of the a86r3 is "548878.0".
the Y of the a86r3 is "1910440.0".

[create a10r4]
a10r4 is a area. "An area (10) in r4".
the printed name of the a10r4 is "Room 4".
Understand "Area 10 in Room 4" as a10r4.
the parent of the a10r4 is r4.
the X of the a10r4 is "549271.3114406404".
the Y of the a10r4 is "1908578.9584036248".

[create a13r4]
a13r4 is a area. "An area (13) in r4".
the printed name of the a13r4 is "Room 4".
Understand "Area 13 in Room 4" as a13r4.
the parent of the a13r4 is r4.
the X of the a13r4 is "550650.6555996799".
the Y of the a13r4 is "1908642.8552974607".

[create a14r4]
a14r4 is a area. "An area (14) in r4".
the printed name of the a14r4 is "Room 4".
Understand "Area 14 in Room 4" as a14r4.
the parent of the a14r4 is r4.
the X of the a14r4 is "550688.6635692203".
the Y of the a14r4 is "1908787.517056374".

[create a19r4]
a19r4 is a area. "An area (19) in r4".
the printed name of the a19r4 is "Room 4".
Understand "Area 19 in Room 4" as a19r4.
the parent of the a19r4 is r4.
the X of the a19r4 is "549645.6022888337".
the Y of the a19r4 is "1908342.131887218".

[create a20r4]
a20r4 is a area. "An area (20) in r4".
the printed name of the a20r4 is "Room 4".
Understand "Area 20 in Room 4" as a20r4.
the parent of the a20r4 is r4.
the X of the a20r4 is "550258.9665046118".
the Y of the a20r4 is "1908338.78788223".

[create a45r4]
a45r4 is a area. "An area (45) in r4".
the printed name of the a45r4 is "Room 4".
Understand "Area 45 in Room 4" as a45r4.
the parent of the a45r4 is r4.
the X of the a45r4 is "550826.2335566806".
the Y of the a45r4 is "1908870.6741779998".

[create a46r4]
a46r4 is a area. "An area (46) in r4".
the printed name of the a46r4 is "Room 4".
Understand "Area 46 in Room 4" as a46r4.
the parent of the a46r4 is r4.
the X of the a46r4 is "550608.5623291694".
the Y of the a46r4 is "1908927.7313395662".

[create a62r4]
a62r4 is a area. "An area (62) in r4".
the printed name of the a62r4 is "Room 4".
Understand "Area 62 in Room 4" as a62r4.
the parent of the a62r4 is r4.
the X of the a62r4 is "549605.0".
the Y of the a62r4 is "1908200.0".

[create a63r4]
a63r4 is a area. "An area (63) in r4".
the printed name of the a63r4 is "Room 4".
Understand "Area 63 in Room 4" as a63r4.
the parent of the a63r4 is r4.
the X of the a63r4 is "549640.5999999999".
the Y of the a63r4 is "1908325.5734335836".

[create a66r4]
a66r4 is a area. "An area (66) in r4".
the printed name of the a66r4 is "Room 4".
Understand "Area 66 in Room 4" as a66r4.
the parent of the a66r4 is r4.
the X of the a66r4 is "549160.0".
the Y of the a66r4 is "1908485.0".

[create a67r4]
a67r4 is a area. "An area (67) in r4".
the printed name of the a67r4 is "Room 4".
Understand "Area 67 in Room 4" as a67r4.
the parent of the a67r4 is r4.
the X of the a67r4 is "549257.3466845369".
the Y of the a67r4 is "1908567.0099861315".

[create a70r4]
a70r4 is a area. "An area (70) in r4".
the printed name of the a70r4 is "Room 4".
Understand "Area 70 in Room 4" as a70r4.
the parent of the a70r4 is r4.
the X of the a70r4 is "550780.0".
the Y of the a70r4 is "1908556.0".

[create a71r4]
a71r4 is a area. "An area (71) in r4".
the printed name of the a71r4 is "Room 4".
Understand "Area 71 in Room 4" as a71r4.
the parent of the a71r4 is r4.
the X of the a71r4 is "550687.9477917007".
the Y of the a71r4 is "1908617.0529526698".

[create a77r4]
a77r4 is a area. "An area (77) in r4".
the printed name of the a77r4 is "Room 4".
Understand "Area 77 in Room 4" as a77r4.
the parent of the a77r4 is r4.
the X of the a77r4 is "550304.0".
the Y of the a77r4 is "1908200.0".

[create a78r4]
a78r4 is a area. "An area (78) in r4".
the printed name of the a78r4 is "Room 4".
Understand "Area 78 in Room 4" as a78r4.
the parent of the a78r4 is r4.
the X of the a78r4 is "550269.7".
the Y of the a78r4 is "1908307.8898876405".

[create a92r4]
a92r4 is a area. "An area (92) in r4".
the printed name of the a92r4 is "Room 4".
Understand "Area 92 in Room 4" as a92r4.
the parent of the a92r4 is r4.
the X of the a92r4 is "550871.0".
the Y of the a92r4 is "1908895.0".

[create a93r4]
a93r4 is a area. "An area (93) in r4".
the printed name of the a93r4 is "Room 4".
Understand "Area 93 in Room 4" as a93r4.
the parent of the a93r4 is r4.
the X of the a93r4 is "550580.0".
the Y of the a93r4 is "1908968.0".

[create a7r5]
a7r5 is a area. "An area (7) in r5".
the printed name of the a7r5 is "Room 5".
Understand "Area 7 in Room 5" as a7r5.
the parent of the a7r5 is r5.
the X of the a7r5 is "545313.8613474357".
the Y of the a7r5 is "1909636.203587741".

[create a26r5]
a26r5 is a area. "An area (26) in r5".
the printed name of the a26r5 is "Room 5".
Understand "Area 26 in Room 5" as a26r5.
the parent of the a26r5 is r5.
the X of the a26r5 is "546719.9051628901".
the Y of the a26r5 is "1909980.7376757741".

[create a27r5]
a27r5 is a area. "An area (27) in r5".
the printed name of the a27r5 is "Room 5".
Understand "Area 27 in Room 5" as a27r5.
the parent of the a27r5 is r5.
the X of the a27r5 is "546773.9818847228".
the Y of the a27r5 is "1909968.5668706093".

[create a28r5]
a28r5 is a area. "An area (28) in r5".
the printed name of the a28r5 is "Room 5".
Understand "Area 28 in Room 5" as a28r5.
the parent of the a28r5 is r5.
the X of the a28r5 is "545522.3937358744".
the Y of the a28r5 is "1910075.5085323553".

[create a29r5]
a29r5 is a area. "An area (29) in r5".
the printed name of the a29r5 is "Room 5".
Understand "Area 29 in Room 5" as a29r5.
the parent of the a29r5 is r5.
the X of the a29r5 is "546519.6767287002".
the Y of the a29r5 is "1910287.363967144".

[create a32r5]
a32r5 is a area. "An area (32) in r5".
the printed name of the a32r5 is "Room 5".
Understand "Area 32 in Room 5" as a32r5.
the parent of the a32r5 is r5.
the X of the a32r5 is "546610.9999999998".
the Y of the a32r5 is "1910404.9925705255".

[create a41r5]
a41r5 is a area. "An area (41) in r5".
the printed name of the a41r5 is "Room 5".
Understand "Area 41 in Room 5" as a41r5.
the parent of the a41r5 is r5.
the X of the a41r5 is "547018.914166946".
the Y of the a41r5 is "1909832.0746232944".

[create a48r5]
a48r5 is a area. "An area (48) in r5".
the printed name of the a48r5 is "Room 5".
Understand "Area 48 in Room 5" as a48r5.
the parent of the a48r5 is r5.
the X of the a48r5 is "545153.0".
the Y of the a48r5 is "1909675.0".

[create a49r5]
a49r5 is a area. "An area (49) in r5".
the printed name of the a49r5 is "Room 5".
Understand "Area 49 in Room 5" as a49r5.
the parent of the a49r5 is r5.
the X of the a49r5 is "545294.3985741445".
the Y of the a49r5 is "1909641.1".

[create a52r5]
a52r5 is a area. "An area (52) in r5".
the printed name of the a52r5 is "Room 5".
Understand "Area 52 in Room 5" as a52r5.
the parent of the a52r5 is r5.
the X of the a52r5 is "546971.0".
the Y of the a52r5 is "1910106.0".

[create a53r5]
a53r5 is a area. "An area (53) in r5".
the printed name of the a53r5 is "Room 5".
Understand "Area 53 in Room 5" as a53r5.
the parent of the a53r5 is r5.
the X of the a53r5 is "546871.3386146746".
the Y of the a53r5 is "1910035.5837845085".

[create a54r5]
a54r5 is a area. "An area (54) in r5".
the printed name of the a54r5 is "Room 5".
Understand "Area 54 in Room 5" as a54r5.
the parent of the a54r5 is r5.
the X of the a54r5 is "545416.0".
the Y of the a54r5 is "1910178.0".

[create a55r5]
a55r5 is a area. "An area (55) in r5".
the printed name of the a55r5 is "Room 5".
Understand "Area 55 in Room 5" as a55r5.
the parent of the a55r5 is r5.
the X of the a55r5 is "545508.5740437233".
the Y of the a55r5 is "1910089.5644165026".

[create a80r5]
a80r5 is a area. "An area (80) in r5".
the printed name of the a80r5 is "Room 5".
Understand "Area 80 in Room 5" as a80r5.
the parent of the a80r5 is r5.
the X of the a80r5 is "547060.0".
the Y of the a80r5 is "1909803.0".

[create a82r5]
a82r5 is a area. "An area (82) in r5".
the printed name of the a82r5 is "Room 5".
Understand "Area 82 in Room 5" as a82r5.
the parent of the a82r5 is r5.
the X of the a82r5 is "546641.0".
the Y of the a82r5 is "1910440.0".

[create d0]
d0 is a door. "A door between a0r1 and a1r2".
d0 is south of a0r1 and north of a1r2.

[create d2]
d2 is a door. "A door between a2r2 and a3r3".
d2 is north of a2r2 and south of a3r3.

[create d4]
d4 is a door. "A door between a5r0 and a6r1".
d4 is north of a5r0 and south of a6r1.

[create d6]
d6 is a door. "A door between a8r0 and a7r5".
d6 is north of a8r0 and south of a7r5.

[create d7]
d7 is a door. "A door between a12r0 and a26r5".
d7 is north of a12r0 and south of a26r5.

[create d8]
d8 is a door. "A door between a9r3 and a10r4".
d8 is south of a9r3 and north of a10r4.

northeast of a4r0 is southwest of a5r0.

north of a11r0 is south of a12r0.

north of a13r4 is south of a14r4.

southeast of a2r2 is northwest of a15r2.

southwest of a1r2 is northeast of a16r2.

southeast of a4r0 is northwest of a17r0.

southeast of a8r0 is northwest of a18r0.

southeast of a10r4 is northwest of a19r4.

southwest of a13r4 is northeast of a20r4.

southwest of a11r0 is northeast of a21r0.

southeast of a11r0 is northwest of a22r0.

southwest of a22r0 is northeast of a23r0.

east of a18r0 is west of a21r0.

east of a19r4 is west of a20r4.

west of a0r1 is east of a6r1.

southeast of a24r3 is northwest of a25r3.

east of a26r5 is west of a27r5.

east of a28r5 is west of a29r5.

east of a30r3 is west of a31r3.

northwest of a26r5 is southeast of a29r5.

northeast of a29r5 is southwest of a32r5.

northeast of a25r3 is southwest of a33r3.

northeast of a0r1 is southwest of a34r1.

northwest of a6r1 is southeast of a35r1.

southwest of a30r3 is northeast of a36r3.

south of a37r3 is north of a38r3.

northeast of a39r3 is southwest of a40r3.

southeast of a27r5 is northwest of a41r5.

southeast of a40r3 is northwest of a42r3.

south of a25r3 is north of a37r3.

north of a40r3 is south of a43r3.

southeast of a31r3 is northwest of a43r3.

northeast of a12r0 is southwest of a44r0.

northeast of a14r4 is southwest of a45r4.

northwest of a14r4 is southeast of a46r4.

north of a3r3 is south of a47r3.

north of a9r3 is south of a36r3.

southwest of a38r3 is northeast of a47r3.

northeast of a7r5 is southwest of a28r5.

east of a48r5 is west of a49r5.

southwest of a50r3 is northeast of a51r3.

southwest of a52r5 is northeast of a53r5.

southeast of a54r5 is northwest of a55r5.

southeast of a56r3 is northwest of a57r3.

south of a58r3 is north of a59r3.

north of a60r0 is south of a61r0.

north of a62r4 is south of a63r4.

northeast of a64r0 is southwest of a65r0.

northeast of a66r4 is southwest of a67r4.

northeast of a68r0 is southwest of a69r0.

northwest of a70r4 is southeast of a71r4.

north of a72r3 is south of a73r3.

north of a72r3 is south of a74r3.

north of a75r0 is south of a76r0.

north of a77r4 is south of a78r4.

southwest of a39r3 is northeast of a79r3.

southeast of a41r5 is northwest of a80r5.

southeast of a42r3 is northwest of a81r3.

northeast of a32r5 is southwest of a82r5.

northeast of a33r3 is southwest of a83r3.

northeast of a34r1 is southwest of a84r1.

northwest of a35r1 is southeast of a85r1.

northwest of a24r3 is southeast of a86r3.

southwest of a23r0 is northeast of a87r0.

southeast of a15r2 is northwest of a88r2.

southwest of a16r2 is northeast of a89r2.

southeast of a17r0 is northwest of a90r0.

northeast of a44r0 is southwest of a91r0.

northeast of a45r4 is southwest of a92r4.

northwest of a46r4 is southeast of a93r4.

east of a3r3 is west of a74r3.

west of a7r5 is east of a49r5.

west of a36r3 is east of a38r3.

northeast of a43r3 is southwest of a51r3.

northeast of a27r5 is southwest of a53r5.

northwest of a28r5 is southeast of a55r5.

northwest of a30r3 is southeast of a57r3.

north of a31r3 is south of a59r3.

south of a19r4 is north of a63r4.

east of a1r2 is west of a2r2.

west of a4r0 is east of a22r0.

west of a5r0 is east of a65r0.

southwest of a10r4 is northeast of a67r4.

southwest of a8r0 is northeast of a69r0.

southeast of a13r4 is northwest of a71r4.

west of a9r3 is east of a73r3.

south of a21r0 is north of a76r0.

south of a20r4 is north of a78r4.

south of a18r0 is north of a61r0.

Definition: a direction (called thatway) is viable if the room thatway from the location is not nowhere.

nLooking is a number that varies. nLooking is 0. 
dirNumber is a number that varies.
dirNumber is 0. 
relDirDesc is a text that varies. 

[describe areas and rooms]
After looking:
	now nLooking is 1; 
	let accessibleRooms be a list of rooms;
	let accessibleAreas be a list of areas;
	let pDirections be list of viable directions;
	let parentSource be the parent of the location of player;
	let relDirs be a list of number;
	repeat with dirToLookAt running through pDirections:
		try silently going dirToLookAt;
		if rule succeeded:
			now dirNumber is 0;
			if "[dirToLookAt]" is "south":
				now dirNumber is 4;
			if "[dirToLookAt]" is "east":
				now dirNumber is 6;
			if "[dirToLookAt]" is "west":
				now dirNumber is 2;
			if "[dirToLookAt]" is "northwest":
				now dirNumber is 1;
			if "[dirToLookAt]" is "southwest":
				now dirNumber is 3;
			if "[dirToLookAt]" is "southeast":
				now dirNumber is 5;
			if "[dirToLookAt]" is "northeast":
				now dirNumber is 7;
			let relDir be the remainder after dividing the orientation of the player - dirNumber + 80 by 8;
			add relDir to relDirs;
			if relDir is 0:
				now relDirDesc is "at the front";
			if relDir is 1:
				now relDirDesc is "at the slight right";
			if relDir is 2:
				now relDirDesc is "at the right";
			if relDir is 3:
				now relDirDesc is "at the sharp right";
			if relDir is 4:
				now relDirDesc is "at the back";
			if relDir is 5:
				now relDirDesc is "at the sharp left";
			if relDir is 6:
				now relDirDesc is "at the left";
			if relDir is 7:
				now relDirDesc is "at the slight left";
			let destinationParent be the parent of the location of the player;
			if "[parentSource]" is "[destinationParent]":
				say "You can continue in the [parentSource] by going [dirToLookAt] ([relDirDesc])[line break]";
				add the location of the player to accessibleAreas;
			otherwise:
				say "You can enter the [destinationParent] by going [dirToLookAt] ([relDirDesc])[line break]";
				add the destinationParent to accessibleRooms;
			try silently going the opposite of dirToLookAt;
	repeat with vO running through the visible_objects of the location:
		say "[vO] is visible from here, but too far! You can move in this room to examine or access [it]";
	now nLooking is 0.
Before going through a locked door when nLooking is 1:
	stop the action.
Going front is an action applying to nothing. Understand "go front" as going front.

Check going front:
	if the orientation of the player is:
		-- 0: try the player going north;
		-- 1: try the player going northwest;
		-- 2: try the player going west;
		-- 3: try the player going southwest;
		-- 4: try the player going south;
		-- 5: try the player going southeast;
		-- 6: try the player going east;
		-- 7: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".


Going back is an action applying to nothing. Understand "go back" as going back.

Check going back:
	if the orientation of the player is:
		-- 4: try the player going north;
		-- 5: try the player going northwest;
		-- 6: try the player going west;
		-- 7: try the player going southwest;
		-- 0: try the player going south;
		-- 1: try the player going southeast;
		-- 2: try the player going east;
		-- 3: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".



Going left is an action applying to nothing. Understand "go left" as going left.

Check going left:
	if the orientation of the player is:
		-- 6: try the player going north;
		-- 7: try the player going northwest;
		-- 0: try the player going west;
		-- 1: try the player going southwest;
		-- 2: try the player going south;
		-- 3: try the player going southeast;
		-- 4: try the player going east;
		-- 5: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".


Going right is an action applying to nothing. Understand "go right" as going right.

Check going right:
	if the orientation of the player is:
		-- 2: try the player going north;
		-- 3: try the player going northwest;
		-- 4: try the player going west;
		-- 5: try the player going southwest;
		-- 6: try the player going south;
		-- 7: try the player going southeast;
		-- 0: try the player going east;
		-- 1: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Going sharp right is an action applying to nothing. Understand "go sharp right" as going sharp right.
Going slight right is an action applying to nothing. Understand "go slight right" as going slight right.
Going sharp left is an action applying to nothing. Understand "go sharp left" as going sharp left.
Going slight left is an action applying to nothing. Understand "go slight left" as going slight left.

Check going sharp right:
	if the orientation of the player is:
		-- 3: try the player going north;
		-- 4: try the player going northwest;
		-- 5: try the player going west;
		-- 6: try the player going southwest;
		-- 7: try the player going south;
		-- 0: try the player going southeast;
		-- 1: try the player going east;
		-- 2: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Check going slight right:
	if the orientation of the player is:
		-- 1: try the player going north;
		-- 2: try the player going northwest;
		-- 3: try the player going west;
		-- 4: try the player going southwest;
		-- 5: try the player going south;
		-- 6: try the player going southeast;
		-- 7: try the player going east;
		-- 0: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Check going sharp left:
	if the orientation of the player is:
		-- 5: try the player going north;
		-- 6: try the player going northwest;
		-- 7: try the player going west;
		-- 0: try the player going southwest;
		-- 1: try the player going south;
		-- 2: try the player going southeast;
		-- 3: try the player going east;
		-- 4: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Check going slight left:
	if the orientation of the player is:
		-- 7: try the player going north;
		-- 0: try the player going northwest;
		-- 1: try the player going west;
		-- 2: try the player going southwest;
		-- 3: try the player going south;
		-- 4: try the player going southeast;
		-- 5: try the player going east;
		-- 6: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".


Instead of going north:
	if nLooking is 0:
		now the orientation of the player is 0;
	continue the action.

Instead of going south:
	if nLooking is 0:
		now the orientation of the player is 4;
	continue the action.

Instead of going east:
	if nLooking is 0:
		now the orientation of the player is 6;
	continue the action.

Instead of going west:
	if nLooking is 0:
		now the orientation of the player is 2;
	continue the action.

Instead of going northwest:
	if nLooking is 0:
		now the orientation of the player is 1;
	continue the action.

Instead of going southwest:
	if nLooking is 0:
		now the orientation of the player is 3;
	continue the action.

Instead of going northeast:
	if nLooking is 0:
		now the orientation of the player is 7;
	continue the action.

Instead of going southeast:
	if nLooking is 0:
		now the orientation of the player is 5;
	continue the action.

Understand "veer left" as going slight left.
Understand "veer right" as going slight right.
Understand "turn left" as going left.
Understand "turn right" as going right.
Understand "turn around" as going back.
Understand "turn sharp left" as going sharp left.
Understand "Go straight" as going front.
Understand "turn sharp right" as going sharp right.
Understand "Turn slightly left" as going slight left.
Understand "Turn slightly right" as going slight right.
Understand "Make a sharp right" as going sharp right.
Understand "Make a sharp left" as going sharp left.
Every turn:
	let xx be the X of the location of the player;
	let yy be the Y of the location of the player;
	say "X: [xx][line break]";
	say "Y: [yy]".
the player is in a53r5.

the orientation of the player is 6.
