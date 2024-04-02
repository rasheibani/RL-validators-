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

[create r6]
r6 is a indoor_room. "Room 6".

[create r7]
r7 is a indoor_room. "Room 7".

[create r8]
r8 is a indoor_room. "Room 8".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "547925.3319092896".
the Y of the a23r0 is "1691729.437845304".

[create a28r0]
a28r0 is a area. "An area (28) in r0".
the printed name of the a28r0 is "Room 0".
Understand "Area 28 in Room 0" as a28r0.
the parent of the a28r0 is r0.
the X of the a28r0 is "548280.8228866081".
the Y of the a28r0 is "1691640.061504006".

[create a30r0]
a30r0 is a area. "An area (30) in r0".
the printed name of the a30r0 is "Room 0".
Understand "Area 30 in Room 0" as a30r0.
the parent of the a30r0 is r0.
the X of the a30r0 is "547596.3913043479".
the Y of the a30r0 is "1691645.5".

[create a34r0]
a34r0 is a area. "An area (34) in r0".
the printed name of the a34r0 is "Room 0".
Understand "Area 34 in Room 0" as a34r0.
the parent of the a34r0 is r0.
the X of the a34r0 is "547549.0".
the Y of the a34r0 is "1691591.0".

[create a35r0]
a35r0 is a area. "An area (35) in r0".
the printed name of the a35r0 is "Room 0".
Understand "Area 35 in Room 0" as a35r0.
the parent of the a35r0 is r0.
the X of the a35r0 is "547579.6227296315".
the Y of the a35r0 is "1691616.5742086142".

[create a38r0]
a38r0 is a area. "An area (38) in r0".
the printed name of the a38r0 is "Room 0".
Understand "Area 38 in Room 0" as a38r0.
the parent of the a38r0 is r0.
the X of the a38r0 is "547540.0".
the Y of the a38r0 is "1691680.0".

[create a39r0]
a39r0 is a area. "An area (39) in r0".
the printed name of the a39r0 is "Room 0".
Understand "Area 39 in Room 0" as a39r0.
the parent of the a39r0 is r0.
the X of the a39r0 is "547582.2240682534".
the Y of the a39r0 is "1691661.792321509".

[create a52r0]
a52r0 is a area. "An area (52) in r0".
the printed name of the a52r0 is "Room 0".
Understand "Area 52 in Room 0" as a52r0.
the parent of the a52r0 is r0.
the X of the a52r0 is "548326.0".
the Y of the a52r0 is "1691591.0".

[create a53r0]
a53r0 is a area. "An area (53) in r0".
the printed name of the a53r0 is "Room 0".
Understand "Area 53 in Room 0" as a53r0.
the parent of the a53r0 is r0.
the X of the a53r0 is "548294.9579422104".
the Y of the a53r0 is "1691616.6166125855".

[create a56r0]
a56r0 is a area. "An area (56) in r0".
the printed name of the a56r0 is "Room 0".
Understand "Area 56 in Room 0" as a56r0.
the parent of the a56r0 is r0.
the X of the a56r0 is "548335.0".
the Y of the a56r0 is "1691680.0".

[create a57r0]
a57r0 is a area. "An area (57) in r0".
the printed name of the a57r0 is "Room 0".
Understand "Area 57 in Room 0" as a57r0.
the parent of the a57r0 is r0.
the X of the a57r0 is "548307.8491587283".
the Y of the a57r0 is "1691660.2680626004".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "548027.8325500068".
the Y of the a2r1 is "1693527.6501174462".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "548122.4999999998".
the Y of the a3r1 is "1693652.784988962".

[create a6r1]
a6r1 is a area. "An area (6) in r1".
the printed name of the a6r1 is "Room 1".
Understand "Area 6 in Room 1" as a6r1.
the parent of the a6r1 is r1.
the X of the a6r1 is "548006.4090909093".
the Y of the a6r1 is "1693552.1212121206".

[create a9r1]
a9r1 is a area. "An area (9) in r1".
the printed name of the a9r1 is "Room 1".
Understand "Area 9 in Room 1" as a9r1.
the parent of the a9r1 is r1.
the X of the a9r1 is "547677.0530944762".
the Y of the a9r1 is "1693579.7325327853".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "547342.4921500895".
the Y of the a11r1 is "1693660.7986347186".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "547652.9091770432".
the Y of the a12r1 is "1693598.236272785".

[create a19r1]
a19r1 is a area. "An area (19) in r1".
the printed name of the a19r1 is "Room 1".
Understand "Area 19 in Room 1" as a19r1.
the parent of the a19r1 is r1.
the X of the a19r1 is "548051.0735241214".
the Y of the a19r1 is "1693411.7267338359".

[create a46r1]
a46r1 is a area. "An area (46) in r1".
the printed name of the a46r1 is "Room 1".
Understand "Area 46 in Room 1" as a46r1.
the parent of the a46r1 is r1.
the X of the a46r1 is "547286.0".
the Y of the a46r1 is "1693624.0".

[create a47r1]
a47r1 is a area. "An area (47) in r1".
the printed name of the a47r1 is "Room 1".
Understand "Area 47 in Room 1" as a47r1.
the parent of the a47r1 is r1.
the X of the a47r1 is "547326.7622699386".
the Y of the a47r1 is "1693642.6487730062".

[create a64r1]
a64r1 is a area. "An area (64) in r1".
the printed name of the a64r1 is "Room 1".
Understand "Area 64 in Room 1" as a64r1.
the parent of the a64r1 is r1.
the X of the a64r1 is "547295.0".
the Y of the a64r1 is "1693714.0".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "547325.9153780068".
the Y of the a65r1 is "1693688.1834621993".

[create a70r1]
a70r1 is a area. "An area (70) in r1".
the printed name of the a70r1 is "Room 1".
Understand "Area 70 in Room 1" as a70r1.
the parent of the a70r1 is r1.
the X of the a70r1 is "547609.0".
the Y of the a70r1 is "1693704.0".

[create a71r1]
a71r1 is a area. "An area (71) in r1".
the printed name of the a71r1 is "Room 1".
Understand "Area 71 in Room 1" as a71r1.
the parent of the a71r1 is r1.
the X of the a71r1 is "547641.994017094".
the Y of the a71r1 is "1693705.7505982905".

[create a72r1]
a72r1 is a area. "An area (72) in r1".
the printed name of the a72r1 is "Room 1".
Understand "Area 72 in Room 1" as a72r1.
the parent of the a72r1 is r1.
the X of the a72r1 is "548054.0".
the Y of the a72r1 is "1693714.0".

[create a73r1]
a73r1 is a area. "An area (73) in r1".
the printed name of the a73r1 is "Room 1".
Understand "Area 73 in Room 1" as a73r1.
the parent of the a73r1 is r1.
the X of the a73r1 is "548022.0059829061".
the Y of the a73r1 is "1693705.7505982907".

[create a81r1]
a81r1 is a area. "An area (81) in r1".
the printed name of the a81r1 is "Room 1".
Understand "Area 81 in Room 1" as a81r1.
the parent of the a81r1 is r1.
the X of the a81r1 is "548145.0".
the Y of the a81r1 is "1693704.0".

[create a13r2]
a13r2 is a area. "An area (13) in r2".
the printed name of the a13r2 is "Room 2".
Understand "Area 13 in Room 2" as a13r2.
the parent of the a13r2 is r2.
the X of the a13r2 is "547166.2896516463".
the Y of the a13r2 is "1691760.8833195833".

[create a17r2]
a17r2 is a area. "An area (17) in r2".
the printed name of the a17r2 is "Room 2".
Understand "Area 17 in Room 2" as a17r2.
the parent of the a17r2 is r2.
the X of the a17r2 is "547144.2390740267".
the Y of the a17r2 is "1691762.431282769".

[create a22r2]
a22r2 is a area. "An area (22) in r2".
the printed name of the a22r2 is "Room 2".
Understand "Area 22 in Room 2" as a22r2.
the parent of the a22r2 is r2.
the X of the a22r2 is "547048.7043713275".
the Y of the a22r2 is "1691622.7067112252".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "547260.4539705467".
the Y of the a26r2 is "1691648.7643334027".

[create a77r2]
a77r2 is a area. "An area (77) in r2".
the printed name of the a77r2 is "Room 2".
Understand "Area 77 in Room 2" as a77r2.
the parent of the a77r2 is r2.
the X of the a77r2 is "547018.0".
the Y of the a77r2 is "1691572.0".

[create a79r2]
a79r2 is a area. "An area (79) in r2".
the printed name of the a79r2 is "Room 2".
Understand "Area 79 in Room 2" as a79r2.
the parent of the a79r2 is r2.
the X of the a79r2 is "547301.0".
the Y of the a79r2 is "1691600.0".

[create a4r3]
a4r3 is a area. "An area (4) in r3".
the printed name of the a4r3 is "Room 3".
Understand "Area 4 in Room 3" as a4r3.
the parent of the a4r3 is r3.
the X of the a4r3 is "546477.3333333333".
the Y of the a4r3 is "1693660.9694444444".

[create a5r3]
a5r3 is a area. "An area (5) in r3".
the printed name of the a5r3 is "Room 3".
Understand "Area 5 in Room 3" as a5r3.
the parent of the a5r3 is r3.
the X of the a5r3 is "546544.5167685037".
the Y of the a5r3 is "1693573.513395759".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "546549.8708448356".
the Y of the a15r3 is "1693405.3162913683".

[create a60r3]
a60r3 is a area. "An area (60) in r3".
the printed name of the a60r3 is "Room 3".
Understand "Area 60 in Room 3" as a60r3.
the parent of the a60r3 is r3.
the X of the a60r3 is "546644.0".
the Y of the a60r3 is "1693704.0".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "546610.6666666667".
the Y of the a61r3 is "1693653.5364495008".

[create a74r3]
a74r3 is a area. "An area (74) in r3".
the printed name of the a74r3 is "Room 3".
Understand "Area 74 in Room 3" as a74r3.
the parent of the a74r3 is r3.
the X of the a74r3 is "546444.0".
the Y of the a74r3 is "1693704.0".

[create a14r4]
a14r4 is a area. "An area (14) in r4".
the printed name of the a14r4 is "Room 4".
Understand "Area 14 in Room 4" as a14r4.
the parent of the a14r4 is r4.
the X of the a14r4 is "545903.7496886394".
the Y of the a14r4 is "1691762.317373606".

[create a16r4]
a16r4 is a area. "An area (16) in r4".
the printed name of the a16r4 is "Room 4".
Understand "Area 16 in Room 4" as a16r4.
the parent of the a16r4 is r4.
the X of the a16r4 is "545890.0086022713".
the Y of the a16r4 is "1691764.0104014392".

[create a21r4]
a21r4 is a area. "An area (21) in r4".
the printed name of the a21r4 is "Room 4".
Understand "Area 21 in Room 4" as a21r4.
the parent of the a21r4 is r4.
the X of the a21r4 is "545786.8816727516".
the Y of the a21r4 is "1691619.4952958587".

[create a24r4]
a24r4 is a area. "An area (24) in r4".
the printed name of the a24r4 is "Room 4".
Understand "Area 24 in Room 4" as a24r4.
the parent of the a24r4 is r4.
the X of the a24r4 is "545997.2968536067".
the Y of the a24r4 is "1691651.4209743899".

[create a76r4]
a76r4 is a area. "An area (76) in r4".
the printed name of the a76r4 is "Room 4".
Understand "Area 76 in Room 4" as a76r4.
the parent of the a76r4 is r4.
the X of the a76r4 is "545756.0".
the Y of the a76r4 is "1691572.0".

[create a78r4]
a78r4 is a area. "An area (78) in r4".
the printed name of the a78r4 is "Room 4".
Understand "Area 78 in Room 4" as a78r4.
the parent of the a78r4 is r4.
the X of the a78r4 is "546038.0".
the Y of the a78r4 is "1691600.0".

[create a7r5]
a7r5 is a area. "An area (7) in r5".
the printed name of the a7r5 is "Room 5".
Understand "Area 7 in Room 5" as a7r5.
the parent of the a7r5 is r5.
the X of the a7r5 is "545050.2813718208".
the Y of the a7r5 is "1693661.2845894436".

[create a8r5]
a8r5 is a area. "An area (8) in r5".
the printed name of the a8r5 is "Room 5".
Understand "Area 8 in Room 5" as a8r5.
the parent of the a8r5 is r5.
the X of the a8r5 is "545405.7977073499".
the Y of the a8r5 is "1693544.5677680378".

[create a10r5]
a10r5 is a area. "An area (10) in r5".
the printed name of the a10r5 is "Room 5".
Understand "Area 10 in Room 5" as a10r5.
the parent of the a10r5 is r5.
the X of the a10r5 is "545779.1653756483".
the Y of the a10r5 is "1693661.2658517454".

[create a48r5]
a48r5 is a area. "An area (48) in r5".
the printed name of the a48r5 is "Room 5".
Understand "Area 48 in Room 5" as a48r5.
the parent of the a48r5 is r5.
the X of the a48r5 is "544994.0".
the Y of the a48r5 is "1693624.0".

[create a49r5]
a49r5 is a area. "An area (49) in r5".
the printed name of the a49r5 is "Room 5".
Understand "Area 49 in Room 5" as a49r5.
the parent of the a49r5 is r5.
the X of the a49r5 is "545033.4941364606".
the Y of the a49r5 is "1693642.775586354".

[create a50r5]
a50r5 is a area. "An area (50) in r5".
the printed name of the a50r5 is "Room 5".
Understand "Area 50 in Room 5" as a50r5.
the parent of the a50r5 is r5.
the X of the a50r5 is "545836.0".
the Y of the a50r5 is "1693624.0".

[create a51r5]
a51r5 is a area. "An area (51) in r5".
the printed name of the a51r5 is "Room 5".
Understand "Area 51 in Room 5" as a51r5.
the parent of the a51r5 is r5.
the X of the a51r5 is "545795.6686930092".
the Y of the a51r5 is "1693642.2965214453".

[create a62r5]
a62r5 is a area. "An area (62) in r5".
the printed name of the a62r5 is "Room 5".
Understand "Area 62 in Room 5" as a62r5.
the parent of the a62r5 is r5.
the X of the a62r5 is "545826.0".
the Y of the a62r5 is "1693714.0".

[create a63r5]
a63r5 is a area. "An area (63) in r5".
the printed name of the a63r5 is "Room 5".
Understand "Area 63 in Room 5" as a63r5.
the parent of the a63r5 is r5.
the X of the a63r5 is "545795.1050372066".
the Y of the a63r5 is "1693687.7894485786".

[create a66r5]
a66r5 is a area. "An area (66) in r5".
the printed name of the a66r5 is "Room 5".
Understand "Area 66 in Room 5" as a66r5.
the parent of the a66r5 is r5.
the X of the a66r5 is "545003.0".
the Y of the a66r5 is "1693714.0".

[create a67r5]
a67r5 is a area. "An area (67) in r5".
the printed name of the a67r5 is "Room 5".
Understand "Area 67 in Room 5" as a67r5.
the parent of the a67r5 is r5.
the X of the a67r5 is "545034.0876068377".
the Y of the a67r5 is "1693688.1662393163".

[create a25r6]
a25r6 is a area. "An area (25) in r6".
the printed name of the a25r6 is "Room 6".
Understand "Area 25 in Room 6" as a25r6.
the parent of the a25r6 is r6.
the X of the a25r6 is "548826.7606867261".
the Y of the a25r6 is "1691749.5866818966".

[create a32r6]
a32r6 is a area. "An area (32) in r6".
the printed name of the a32r6 is "Room 6".
Understand "Area 32 in Room 6" as a32r6.
the parent of the a32r6 is r6.
the X of the a32r6 is "548761.5661679944".
the Y of the a32r6 is "1691619.934159247".

[create a33r6]
a33r6 is a area. "An area (33) in r6".
the printed name of the a33r6 is "Room 6".
Understand "Area 33 in Room 6" as a33r6.
the parent of the a33r6 is r6.
the X of the a33r6 is "548824.0091290362".
the Y of the a33r6 is "1691730.1696830264".

[create a42r6]
a42r6 is a area. "An area (42) in r6".
the printed name of the a42r6 is "Room 6".
Understand "Area 42 in Room 6" as a42r6.
the parent of the a42r6 is r6.
the X of the a42r6 is "548928.0".
the Y of the a42r6 is "1691603.0".

[create a43r6]
a43r6 is a area. "An area (43) in r6".
the printed name of the a43r6 is "Room 6".
Understand "Area 43 in Room 6" as a43r6.
the parent of the a43r6 is r6.
the X of the a43r6 is "548884.8503889911".
the Y of the a43r6 is "1691661.5027281872".

[create a75r6]
a75r6 is a area. "An area (75) in r6".
the printed name of the a75r6 is "Room 6".
Understand "Area 75 in Room 6" as a75r6.
the parent of the a75r6 is r6.
the X of the a75r6 is "548738.0".
the Y of the a75r6 is "1691569.0".

[create a0r7]
a0r7 is a area. "An area (0) in r7".
the printed name of the a0r7 is "Room 7".
Understand "Area 0 in Room 7" as a0r7.
the parent of the a0r7 is r7.
the X of the a0r7 is "549642.0".
the Y of the a0r7 is "1693647.7422185428".

[create a1r7]
a1r7 is a area. "An area (1) in r7".
the printed name of the a1r7 is "Room 7".
Understand "Area 1 in Room 7" as a1r7.
the parent of the a1r7 is r7.
the X of the a1r7 is "549726.6187909032".
the Y of the a1r7 is "1693523.5461257002".

[create a18r7]
a18r7 is a area. "An area (18) in r7".
the printed name of the a18r7 is "Room 7".
Understand "Area 18 in Room 7" as a18r7.
the parent of the a18r7 is r7.
the X of the a18r7 is "549697.9001965041".
the Y of the a18r7 is "1693393.9469272913".

[create a20r7]
a20r7 is a area. "An area (20) in r7".
the printed name of the a20r7 is "Room 7".
Understand "Area 20 in Room 7" as a20r7.
the parent of the a20r7 is r7.
the X of the a20r7 is "550102.8606168446".
the Y of the a20r7 is "1693659.860290629".

[create a44r7]
a44r7 is a area. "An area (44) in r7".
the printed name of the a44r7 is "Room 7".
Understand "Area 44 in Room 7" as a44r7.
the parent of the a44r7 is r7.
the X of the a44r7 is "550160.0".
the Y of the a44r7 is "1693624.0".

[create a45r7]
a45r7 is a area. "An area (45) in r7".
the printed name of the a45r7 is "Room 7".
Understand "Area 45 in Room 7" as a45r7.
the parent of the a45r7 is r7.
the X of the a45r7 is "550117.9400000001".
the Y of the a45r7 is "1693642.519".

[create a68r7]
a68r7 is a area. "An area (68) in r7".
the printed name of the a68r7 is "Room 7".
Understand "Area 68 in Room 7" as a68r7.
the parent of the a68r7 is r7.
the X of the a68r7 is "550151.0".
the Y of the a68r7 is "1693714.0".

[create a69r7]
a69r7 is a area. "An area (69) in r7".
the printed name of the a69r7 is "Room 7".
Understand "Area 69 in Room 7" as a69r7.
the parent of the a69r7 is r7.
the X of the a69r7 is "550119.9123931624".
the Y of the a69r7 is "1693688.1662393163".

[create a80r7]
a80r7 is a area. "An area (80) in r7".
the printed name of the a80r7 is "Room 7".
Understand "Area 80 in Room 7" as a80r7.
the parent of the a80r7 is r7.
the X of the a80r7 is "549607.0".
the Y of the a80r7 is "1693704.0".

[create a27r8]
a27r8 is a area. "An area (27) in r8".
the printed name of the a27r8 is "Room 8".
Understand "Area 27 in Room 8" as a27r8.
the parent of the a27r8 is r8.
the X of the a27r8 is "549807.0668298497".
the Y of the a27r8 is "1691765.895836697".

[create a29r8]
a29r8 is a area. "An area (29) in r8".
the printed name of the a29r8 is "Room 8".
Understand "Area 29 in Room 8" as a29r8.
the parent of the a29r8 is r8.
the X of the a29r8 is "550170.2584382973".
the Y of the a29r8 is "1691644.9474149994".

[create a31r8]
a31r8 is a area. "An area (31) in r8".
the printed name of the a31r8 is "Room 8".
Understand "Area 31 in Room 8" as a31r8.
the parent of the a31r8 is r8.
the X of the a31r8 is "549421.9132133438".
the Y of the a31r8 is "1691640.2221422398".

[create a36r8]
a36r8 is a area. "An area (36) in r8".
the printed name of the a36r8 is "Room 8".
Understand "Area 36 in Room 8" as a36r8.
the parent of the a36r8 is r8.
the X of the a36r8 is "550218.0".
the Y of the a36r8 is "1691591.0".

[create a37r8]
a37r8 is a area. "An area (37) in r8".
the printed name of the a37r8 is "Room 8".
Understand "Area 37 in Room 8" as a37r8.
the parent of the a37r8 is r8.
the X of the a37r8 is "550186.9765979422".
the Y of the a37r8 is "1691617.0166743887".

[create a40r8]
a40r8 is a area. "An area (40) in r8".
the printed name of the a40r8 is "Room 8".
Understand "Area 40 in Room 8" as a40r8.
the parent of the a40r8 is r8.
the X of the a40r8 is "550228.0".
the Y of the a40r8 is "1691680.0".

[create a41r8]
a41r8 is a area. "An area (41) in r8".
the printed name of the a41r8 is "Room 8".
Understand "Area 41 in Room 8" as a41r8.
the parent of the a41r8 is r8.
the X of the a41r8 is "550185.024710888".
the Y of the a41r8 is "1691662.2977852935".

[create a54r8]
a54r8 is a area. "An area (54) in r8".
the printed name of the a54r8 is "Room 8".
Understand "Area 54 in Room 8" as a54r8.
the parent of the a54r8 is r8.
the X of the a54r8 is "549377.0".
the Y of the a54r8 is "1691591.0".

[create a55r8]
a55r8 is a area. "An area (55) in r8".
the printed name of the a55r8 is "Room 8".
Understand "Area 55 in Room 8" as a55r8.
the parent of the a55r8 is r8.
the X of the a55r8 is "549408.0234020579".
the Y of the a55r8 is "1691617.0166743887".

[create a58r8]
a58r8 is a area. "An area (58) in r8".
the printed name of the a58r8 is "Room 8".
Understand "Area 58 in Room 8" as a58r8.
the parent of the a58r8 is r8.
the X of the a58r8 is "549367.0".
the Y of the a58r8 is "1691680.0".

[create a59r8]
a59r8 is a area. "An area (59) in r8".
the printed name of the a59r8 is "Room 8".
Understand "Area 59 in Room 8" as a59r8.
the parent of the a59r8 is r8.
the X of the a59r8 is "549398.3505784648".
the Y of the a59r8 is "1691660.9916380297".

[create d0]
d0 is a door. "A door between a23r0 and a19r1".
d0 is north of a23r0 and south of a19r1.

[create d2]
d2 is a door. "A door between a25r6 and a18r7".
d2 is northeast of a25r6 and southwest of a18r7.

[create d4]
d4 is a door. "A door between a19r1 and a25r6".
d4 is southeast of a19r1 and northwest of a25r6.

[create d6]
d6 is a door. "A door between a18r7 and a27r8".
d6 is south of a18r7 and north of a27r8.

[create d8]
d8 is a door. "A door between a9r1 and a13r2".
d8 is south of a9r1 and north of a13r2.

[create d10]
d10 is a door. "A door between a15r3 and a14r4".
d10 is south of a15r3 and north of a14r4.

[create d12]
d12 is a door. "A door between a16r4 and a8r5".
d12 is north of a16r4 and south of a8r5.

[create d14]
d14 is a door. "A door between a17r2 and a15r3".
d14 is northwest of a17r2 and southeast of a15r3.

southeast of a0r7 is northwest of a1r7.

northeast of a2r1 is southwest of a3r1.

southeast of a4r3 is northwest of a5r3.

northwest of a2r1 is southeast of a6r1.

east of a7r5 is west of a8r5.

west of a6r1 is east of a9r1.

east of a8r5 is west of a10r5.

east of a11r1 is west of a12r1.

south of a1r7 is north of a18r7.

south of a5r3 is north of a15r3.

south of a2r1 is north of a19r1.

east of a1r7 is west of a20r7.

southwest of a16r4 is northeast of a21r4.

southwest of a17r2 is northeast of a22r2.

southeast of a14r4 is northwest of a24r4.

southeast of a13r2 is northwest of a26r2.

east of a23r0 is west of a28r0.

east of a27r8 is west of a29r8.

west of a23r0 is east of a30r0.

west of a27r8 is east of a31r8.

northeast of a32r6 is southwest of a33r6.

northeast of a34r0 is southwest of a35r0.

northwest of a36r8 is southeast of a37r8.

southeast of a38r0 is northwest of a39r0.

west of a40r8 is east of a41r8.

northwest of a42r6 is southeast of a43r6.

northwest of a44r7 is southeast of a45r7.

northeast of a46r1 is southwest of a47r1.

northeast of a48r5 is southwest of a49r5.

northwest of a50r5 is southeast of a51r5.

northwest of a52r0 is southeast of a53r0.

northeast of a54r8 is southwest of a55r8.

southwest of a56r0 is northeast of a57r0.

southeast of a58r8 is northwest of a59r8.

southwest of a60r3 is northeast of a61r3.

southwest of a62r5 is northeast of a63r5.

southeast of a64r1 is northwest of a65r1.

southeast of a66r5 is northwest of a67r5.

southwest of a68r7 is northeast of a69r7.

east of a70r1 is west of a71r1.

west of a72r1 is east of a73r1.

northwest of a4r3 is southeast of a74r3.

southwest of a32r6 is northeast of a75r6.

southwest of a21r4 is northeast of a76r4.

southwest of a22r2 is northeast of a77r2.

southeast of a24r4 is northwest of a78r4.

southeast of a26r2 is northwest of a79r2.

northwest of a0r7 is southeast of a80r7.

northeast of a3r1 is southwest of a81r1.

northeast of a29r8 is southwest of a41r8.

southeast of a33r6 is northwest of a43r6.

south of a25r6 is north of a33r6.

west of a13r2 is east of a17r2.

west of a14r4 is east of a16r4.

northwest of a9r1 is southeast of a12r1.

southeast of a20r7 is northwest of a45r7.

southwest of a11r1 is northeast of a47r1.

southwest of a7r5 is northeast of a49r5.

southeast of a28r0 is northwest of a53r0.

southwest of a31r8 is northeast of a55r8.

northeast of a28r0 is southwest of a57r0.

northwest of a31r8 is southeast of a59r8.

southwest of a30r0 is northeast of a35r0.

southeast of a29r8 is northwest of a37r8.

northwest of a30r0 is southeast of a39r0.

north of a6r1 is south of a73r1.

southeast of a10r5 is northwest of a51r5.

northeast of a5r3 is southwest of a61r3.

northeast of a10r5 is southwest of a63r5.

northwest of a11r1 is southeast of a65r1.

northwest of a7r5 is southeast of a67r5.

northeast of a20r7 is southwest of a69r7.

north of a12r1 is south of a71r1.

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
the player is in a9r1.

the orientation of the player is 6.
