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
the X of the a4r0 is "547017.2392504469".
the Y of the a4r0 is "2434586.0258407197".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "547123.4999999998".
the Y of the a5r0 is "2434773.3544838177".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "546709.166666667".
the Y of the a8r0 is "2434802.5".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "546953.0867006043".
the Y of the a9r0 is "2434566.2409004136".

[create a45r0]
a45r0 is a area. "An area (45) in r0".
the printed name of the a45r0 is "Room 0".
Understand "Area 45 in Room 0" as a45r0.
the parent of the a45r0 is r0.
the X of the a45r0 is "547154.0".
the Y of the a45r0 is "2434840.0".

[create a47r0]
a47r0 is a area. "An area (47) in r0".
the printed name of the a47r0 is "Room 0".
Understand "Area 47 in Room 0" as a47r0.
the parent of the a47r0 is r0.
the X of the a47r0 is "546680.0".
the Y of the a47r0 is "2434840.0".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "548371.551876286".
the Y of the a14r1 is "2432870.965684384".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "548394.675203877".
the Y of the a15r1 is "2432794.300498058".

[create a25r1]
a25r1 is a area. "An area (25) in r1".
the printed name of the a25r1 is "Room 1".
Understand "Area 25 in Room 1" as a25r1.
the parent of the a25r1 is r1.
the X of the a25r1 is "548550.625".
the Y of the a25r1 is "2432639.4421657096".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "548289.0".
the Y of the a34r1 is "2432600.0".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "548326.375".
the Y of the a35r1 is "2432671.1563946763".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "548588.0".
the Y of the a44r1 is "2432600.0".

[create a2r2]
a2r2 is a area. "An area (2) in r2".
the printed name of the a2r2 is "Room 2".
Understand "Area 2 in Room 2" as a2r2.
the parent of the a2r2 is r2.
the X of the a2r2 is "548546.66386699".
the Y of the a2r2 is "2434626.1110704765".

[create a6r2]
a6r2 is a area. "An area (6) in r2".
the printed name of the a6r2 is "Room 2".
Understand "Area 6 in Room 2" as a6r2.
the parent of the a6r2 is r2.
the X of the a6r2 is "548343.875".
the Y of the a6r2 is "2434837.7358490564".

[create a7r2]
a7r2 is a area. "An area (7) in r2".
the printed name of the a7r2 is "Room 2".
Understand "Area 7 in Room 2" as a7r2.
the parent of the a7r2 is r2.
the X of the a7r2 is "548525.8390816163".
the Y of the a7r2 is "2434622.6986580063".

[create a10r2]
a10r2 is a area. "An area (10) in r2".
the printed name of the a10r2 is "Room 2".
Understand "Area 10 in Room 2" as a10r2.
the parent of the a10r2 is r2.
the X of the a10r2 is "548623.9999999998".
the Y of the a10r2 is "2434769.0265243906".

[create a11r2]
a11r2 is a area. "An area (11) in r2".
the printed name of the a11r2 is "Room 2".
Understand "Area 11 in Room 2" as a11r2.
the parent of the a11r2 is r2.
the X of the a11r2 is "550058.7788069451".
the Y of the a11r2 is "2434699.0165580204".

[create a12r2]
a12r2 is a area. "An area (12) in r2".
the printed name of the a12r2 is "Room 2".
Understand "Area 12 in Room 2" as a12r2.
the parent of the a12r2 is r2.
the X of the a12r2 is "550288.0999999999".
the Y of the a12r2 is "2434816.2047460936".

[create a30r2]
a30r2 is a area. "An area (30) in r2".
the printed name of the a30r2 is "Room 2".
Understand "Area 30 in Room 2" as a30r2.
the parent of the a30r2 is r2.
the X of the a30r2 is "549983.0".
the Y of the a30r2 is "2434840.0".

[create a31r2]
a31r2 is a area. "An area (31) in r2".
the printed name of the a31r2 is "Room 2".
Understand "Area 31 in Room 2" as a31r2.
the parent of the a31r2 is r2.
the X of the a31r2 is "550016.9000000001".
the Y of the a31r2 is "2434771.2575000003".

[create a46r2]
a46r2 is a area. "An area (46) in r2".
the printed name of the a46r2 is "Room 2".
Understand "Area 46 in Room 2" as a46r2.
the parent of the a46r2 is r2.
the X of the a46r2 is "548309.0".
the Y of the a46r2 is "2434835.4716981133".

[create a48r2]
a48r2 is a area. "An area (48) in r2".
the printed name of the a48r2 is "Room 2".
Understand "Area 48 in Room 2" as a48r2.
the parent of the a48r2 is r2.
the X of the a48r2 is "548588.0".
the Y of the a48r2 is "2434840.0".

[create a49r2]
a49r2 is a area. "An area (49) in r2".
the printed name of the a49r2 is "Room 2".
Understand "Area 49 in Room 2" as a49r2.
the parent of the a49r2 is r2.
the X of the a49r2 is "550322.0".
the Y of the a49r2 is "2434840.0".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "549299.5".
the Y of the a3r3 is "2433647.25".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "549190.375".
the Y of the a22r3 is "2432636.923076923".

[create a23r3]
a23r3 is a area. "An area (23) in r3".
the printed name of the a23r3 is "Room 3".
Understand "Area 23 in Room 3" as a23r3.
the parent of the a23r3 is r3.
the X of the a23r3 is "549299.5000000002".
the Y of the a23r3 is "2432745.516271033".

[create a24r3]
a24r3 is a area. "An area (24) in r3".
the printed name of the a24r3 is "Room 3".
Understand "Area 24 in Room 3" as a24r3.
the parent of the a24r3 is r3.
the X of the a24r3 is "549408.625".
the Y of the a24r3 is "2432636.923076923".

[create a42r3]
a42r3 is a area. "An area (42) in r3".
the printed name of the a42r3 is "Room 3".
Understand "Area 42 in Room 3" as a42r3.
the parent of the a42r3 is r3.
the X of the a42r3 is "549154.0".
the Y of the a42r3 is "2432600.0".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "549445.0".
the Y of the a43r3 is "2432600.0".

[create a16r4]
a16r4 is a area. "An area (16) in r4".
the printed name of the a16r4 is "Room 4".
Understand "Area 16 in Room 4" as a16r4.
the parent of the a16r4 is r4.
the X of the a16r4 is "546992.5730620159".
the Y of the a16r4 is "2432900.0599999996".

[create a17r4]
a17r4 is a area. "An area (17) in r4".
the printed name of the a17r4 is "Room 4".
Understand "Area 17 in Room 4" as a17r4.
the parent of the a17r4 is r4.
the X of the a17r4 is "546907.3513579306".
the Y of the a17r4 is "2433154.744676784".

[create a18r4]
a18r4 is a area. "An area (18) in r4".
the printed name of the a18r4 is "Room 4".
Understand "Area 18 in Room 4" as a18r4.
the parent of the a18r4 is r4.
the X of the a18r4 is "546696.4989356409".
the Y of the a18r4 is "2432943.715754728".

[create a19r4]
a19r4 is a area. "An area (19) in r4".
the printed name of the a19r4 is "Room 4".
Understand "Area 19 in Room 4" as a19r4.
the parent of the a19r4 is r4.
the X of the a19r4 is "545523.5279517956".
the Y of the a19r4 is "2432915.0924056573".

[create a20r4]
a20r4 is a area. "An area (20) in r4".
the printed name of the a20r4 is "Room 4".
Understand "Area 20 in Room 4" as a20r4.
the parent of the a20r4 is r4.
the X of the a20r4 is "546994.733923032".
the Y of the a20r4 is "2432739.8860382643".

[create a21r4]
a21r4 is a area. "An area (21) in r4".
the printed name of the a21r4 is "Room 4".
Understand "Area 21 in Room 4" as a21r4.
the parent of the a21r4 is r4.
the X of the a21r4 is "547099.125".
the Y of the a21r4 is "2432638.260869565".

[create a26r4]
a26r4 is a area. "An area (26) in r4".
the printed name of the a26r4 is "Room 4".
Understand "Area 26 in Room 4" as a26r4.
the parent of the a26r4 is r4.
the X of the a26r4 is "546397.1952968487".
the Y of the a26r4 is "2432726.3355115154".

[create a27r4]
a27r4 is a area. "An area (27) in r4".
the printed name of the a27r4 is "Room 4".
Understand "Area 27 in Room 4" as a27r4.
the parent of the a27r4 is r4.
the X of the a27r4 is "546889.875".
the Y of the a27r4 is "2432633.34".

[create a28r4]
a28r4 is a area. "An area (28) in r4".
the printed name of the a28r4 is "Room 4".
Understand "Area 28 in Room 4" as a28r4.
the parent of the a28r4 is r4.
the X of the a28r4 is "546821.0".
the Y of the a28r4 is "2432845.0".

[create a29r4]
a29r4 is a area. "An area (29) in r4".
the printed name of the a29r4 is "Room 4".
Understand "Area 29 in Room 4" as a29r4.
the parent of the a29r4 is r4.
the X of the a29r4 is "546740.2718593397".
the Y of the a29r4 is "2432901.418515639".

[create a32r4]
a32r4 is a area. "An area (32) in r4".
the printed name of the a32r4 is "Room 4".
Understand "Area 32 in Room 4" as a32r4.
the parent of the a32r4 is r4.
the X of the a32r4 is "546430.0".
the Y of the a32r4 is "2432600.0".

[create a33r4]
a33r4 is a area. "An area (33) in r4".
the printed name of the a33r4 is "Room 4".
Understand "Area 33 in Room 4" as a33r4.
the parent of the a33r4 is r4.
the X of the a33r4 is "546398.2".
the Y of the a33r4 is "2432723.1672108844".

[create a36r4]
a36r4 is a area. "An area (36) in r4".
the printed name of the a36r4 is "Room 4".
Understand "Area 36 in Room 4" as a36r4.
the parent of the a36r4 is r4.
the X of the a36r4 is "546855.0".
the Y of the a36r4 is "2432933.4".

[create a37r4]
a37r4 is a area. "An area (37) in r4".
the printed name of the a37r4 is "Room 4".
Understand "Area 37 in Room 4" as a37r4.
the parent of the a37r4 is r4.
the X of the a37r4 is "546875.7240000004".
the Y of the a37r4 is "2432900.06".

[create a38r4]
a38r4 is a area. "An area (38) in r4".
the printed name of the a38r4 is "Room 4".
Understand "Area 38 in Room 4" as a38r4.
the parent of the a38r4 is r4.
the X of the a38r4 is "545410.0".
the Y of the a38r4 is "2432813.0".

[create a39r4]
a39r4 is a area. "An area (39) in r4".
the printed name of the a39r4 is "Room 4".
Understand "Area 39 in Room 4" as a39r4.
the parent of the a39r4 is r4.
the X of the a39r4 is "545492.3272677008".
the Y of the a39r4 is "2432886.7235502647".

[create a40r4]
a40r4 is a area. "An area (40) in r4".
the printed name of the a40r4 is "Room 4".
Understand "Area 40 in Room 4" as a40r4.
the parent of the a40r4 is r4.
the X of the a40r4 is "546855.0".
the Y of the a40r4 is "2432600.0".

[create a41r4]
a41r4 is a area. "An area (41) in r4".
the printed name of the a41r4 is "Room 4".
Understand "Area 41 in Room 4" as a41r4.
the parent of the a41r4 is r4.
the X of the a41r4 is "547134.0".
the Y of the a41r4 is "2432600.0".

[create a0r5]
a0r5 is a area. "An area (0) in r5".
the printed name of the a0r5 is "Room 5".
Understand "Area 0 in Room 5" as a0r5.
the parent of the a0r5 is r5.
the X of the a0r5 is "545278.3750000001".
the Y of the a0r5 is "2434802.5882352944".

[create a1r5]
a1r5 is a area. "An area (1) in r5".
the printed name of the a1r5 is "Room 5".
Understand "Area 1 in Room 5" as a1r5.
the parent of the a1r5 is r5.
the X of the a1r5 is "545387.5051680319".
the Y of the a1r5 is "2434694.437512991".

[create a13r5]
a13r5 is a area. "An area (13) in r5".
the printed name of the a13r5 is "Room 5".
Understand "Area 13 in Room 5" as a13r5.
the parent of the a13r5 is r5.
the X of the a13r5 is "545496.625".
the Y of the a13r5 is "2434802.5".

[create a50r5]
a50r5 is a area. "An area (50) in r5".
the printed name of the a50r5 is "Room 5".
Understand "Area 50 in Room 5" as a50r5.
the parent of the a50r5 is r5.
the X of the a50r5 is "545533.0".
the Y of the a50r5 is "2434840.0".

[create a51r5]
a51r5 is a area. "An area (51) in r5".
the printed name of the a51r5 is "Room 5".
Understand "Area 51 in Room 5" as a51r5.
the parent of the a51r5 is r5.
the X of the a51r5 is "545242.0".
the Y of the a51r5 is "2434840.0".

[create d0]
d0 is a door. "A door between a4r0 and a14r1".
d0 is southeast of a4r0 and northwest of a14r1.

[create d2]
d2 is a door. "A door between a14r1 and a7r2".
d2 is north of a14r1 and south of a7r2.

[create d4]
d4 is a door. "A door between a19r4 and a1r5".
d4 is north of a19r4 and south of a1r5.

[create d6]
d6 is a door. "A door between a9r0 and a17r4".
d6 is south of a9r0 and north of a17r4.

[create d8]
d8 is a door. "A door between a2r2 and a3r3".
d8 is southeast of a2r2 and northwest of a3r3.

[create d9]
d9 is a door. "A door between a11r2 and a3r3".
d9 is southwest of a11r2 and northeast of a3r3.

southeast of a0r5 is northwest of a1r5.

northeast of a4r0 is southwest of a5r0.

southeast of a6r2 is northwest of a7r2.

southeast of a8r0 is northwest of a9r0.

west of a4r0 is east of a9r0.

northeast of a2r2 is southwest of a10r2.

northeast of a11r2 is southwest of a12r2.

northeast of a1r5 is southwest of a13r5.

south of a14r1 is north of a15r1.

north of a16r4 is south of a17r4.

southwest of a17r4 is northeast of a18r4.

southeast of a20r4 is northwest of a21r4.

northeast of a22r3 is southwest of a23r3.

southeast of a23r3 is northwest of a24r3.

southeast of a15r1 is northwest of a25r1.

east of a19r4 is west of a26r4.

southwest of a18r4 is northeast of a26r4.

south of a16r4 is north of a20r4.

south of a3r3 is north of a23r3.

southwest of a20r4 is northeast of a27r4.

northwest of a28r4 is southeast of a29r4.

southeast of a30r2 is northwest of a31r2.

north of a32r4 is south of a33r4.

northeast of a34r1 is southwest of a35r1.

southeast of a36r4 is northwest of a37r4.

northeast of a38r4 is southwest of a39r4.

southwest of a27r4 is northeast of a40r4.

southeast of a21r4 is northwest of a41r4.

southwest of a22r3 is northeast of a42r3.

southeast of a24r3 is northwest of a43r3.

southeast of a25r1 is northwest of a44r1.

northeast of a5r0 is southwest of a45r0.

west of a6r2 is east of a46r2.

northwest of a8r0 is southeast of a47r0.

northwest of a10r2 is southeast of a48r2.

northeast of a12r2 is southwest of a49r2.

northeast of a13r5 is southwest of a50r5.

northwest of a0r5 is southeast of a51r5.

southeast of a18r4 is northwest of a29r4.

west of a2r2 is east of a7r2.

northwest of a11r2 is southeast of a31r2.

south of a26r4 is north of a33r4.

southwest of a15r1 is northeast of a35r1.

west of a16r4 is east of a37r4.

southwest of a19r4 is northeast of a39r4.

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
the player is in a23r3.

the orientation of the player is 6.
