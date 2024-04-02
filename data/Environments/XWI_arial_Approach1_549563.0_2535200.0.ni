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

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "546725.5078295618".
the Y of the a4r0 is "2537281.6406066758".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "546940.5".
the Y of the a5r0 is "2537414.379081815".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "546578.75".
the Y of the a10r0 is "2537420.4929351737".

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "546710.4604242069".
the Y of the a12r0 is "2537205.0055419225".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "545386.2064445316".
the Y of the a16r0 is "2537295.670986919".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "546027.685593508".
the Y of the a17r0 is "2536430.6292565824".

[create a18r0]
a18r0 is a area. "An area (18) in r0".
the printed name of the a18r0 is "Room 0".
Understand "Area 18 in Room 0" as a18r0.
the parent of the a18r0 is r0.
the X of the a18r0 is "545148.7".
the Y of the a18r0 is "2537417.2770407326".

[create a31r0]
a31r0 is a area. "An area (31) in r0".
the printed name of the a31r0 is "Room 0".
Understand "Area 31 in Room 0" as a31r0.
the parent of the a31r0 is r0.
the X of the a31r0 is "545461.0".
the Y of the a31r0 is "2537440.0".

[create a32r0]
a32r0 is a area. "An area (32) in r0".
the printed name of the a32r0 is "Room 0".
Understand "Area 32 in Room 0" as a32r0.
the parent of the a32r0 is r0.
the X of the a32r0 is "545426.2999999999".
the Y of the a32r0 is "2537368.436524823".

[create a37r0]
a37r0 is a area. "An area (37) in r0".
the printed name of the a37r0 is "Room 0".
Understand "Area 37 in Room 0" as a37r0.
the parent of the a37r0 is r0.
the X of the a37r0 is "546970.0".
the Y of the a37r0 is "2537440.0".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "546554.0".
the Y of the a40r0 is "2537440.0".

[create a46r0]
a46r0 is a area. "An area (46) in r0".
the printed name of the a46r0 is "Room 0".
Understand "Area 46 in Room 0" as a46r0.
the parent of the a46r0 is r0.
the X of the a46r0 is "545114.0".
the Y of the a46r0 is "2537440.0".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "547254.1566116107".
the Y of the a13r1 is "2535353.4894037787".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "547316.9412801262".
the Y of the a15r1 is "2535354.610558574".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "547174.125".
the Y of the a23r1 is "2535250.1645021765".

[create a24r1]
a24r1 is a area. "An area (24) in r1".
the printed name of the a24r1 is "Room 1".
Understand "Area 24 in Room 1" as a24r1.
the parent of the a24r1 is r1.
the X of the a24r1 is "547396.875".
the Y of the a24r1 is "2535250.099474627".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "547137.0".
the Y of the a44r1 is "2535200.0".

[create a45r1]
a45r1 is a area. "An area (45) in r1".
the printed name of the a45r1 is "Room 1".
Understand "Area 45 in Room 1" as a45r1.
the parent of the a45r1 is r1.
the X of the a45r1 is "547434.0".
the Y of the a45r1 is "2535200.0".

[create a6r2]
a6r2 is a area. "An area (6) in r2".
the printed name of the a6r2 is "Room 2".
Understand "Area 6 in Room 2" as a6r2.
the parent of the a6r2 is r2.
the X of the a6r2 is "548039.8305982075".
the Y of the a6r2 is "2537298.8947038352".

[create a7r2]
a7r2 is a area. "An area (7) in r2".
the printed name of the a7r2 is "Room 2".
Understand "Area 7 in Room 2" as a7r2.
the parent of the a7r2 is r2.
the X of the a7r2 is "548112.9000000001".
the Y of the a7r2 is "2537389.466754617".

[create a8r2]
a8r2 is a area. "An area (8) in r2".
the printed name of the a8r2 is "Room 2".
Understand "Area 8 in Room 2" as a8r2.
the parent of the a8r2 is r2.
the X of the a8r2 is "547832.1".
the Y of the a8r2 is "2537389.523978247".

[create a9r2]
a9r2 is a area. "An area (9) in r2".
the printed name of the a9r2 is "Room 2".
Understand "Area 9 in Room 2" as a9r2.
the parent of the a9r2 is r2.
the X of the a9r2 is "547903.5565103929".
the Y of the a9r2 is "2537302.5378856105".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "548148.0".
the Y of the a38r2 is "2537440.0".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "547797.0".
the Y of the a39r2 is "2537440.0".

[create a11r3]
a11r3 is a area. "An area (11) in r3".
the printed name of the a11r3 is "Room 3".
Understand "Area 11 in Room 3" as a11r3.
the parent of the a11r3 is r3.
the X of the a11r3 is "548661.5785148891".
the Y of the a11r3 is "2535354.332188505".

[create a20r3]
a20r3 is a area. "An area (20) in r3".
the printed name of the a20r3 is "Room 3".
Understand "Area 20 in Room 3" as a20r3.
the parent of the a20r3 is r3.
the X of the a20r3 is "548533.0".
the Y of the a20r3 is "2535249.422623434".

[create a21r3]
a21r3 is a area. "An area (21) in r3".
the printed name of the a21r3 is "Room 3".
Understand "Area 21 in Room 3" as a21r3.
the parent of the a21r3 is r3.
the X of the a21r3 is "548613.2576810808".
the Y of the a21r3 is "2535353.704645209".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "548743.0".
the Y of the a22r3 is "2535248.7833599267".

[create a42r3]
a42r3 is a area. "An area (42) in r3".
the printed name of the a42r3 is "Room 3".
Understand "Area 42 in Room 3" as a42r3.
the parent of the a42r3 is r3.
the X of the a42r3 is "548498.0".
the Y of the a42r3 is "2535200.0".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "548778.0".
the Y of the a43r3 is "2535200.0".

[create a0r4]
a0r4 is a area. "An area (0) in r4".
the printed name of the a0r4 is "Room 4".
Understand "Area 0 in Room 4" as a0r4.
the parent of the a0r4 is r4.
the X of the a0r4 is "549352.8389889363".
the Y of the a0r4 is "2537230.3326230906".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "549532.6666666666".
the Y of the a1r4 is "2537401.379310345".

[create a2r4]
a2r4 is a area. "An area (2) in r4".
the printed name of the a2r4 is "Room 4".
Understand "Area 2 in Room 4" as a2r4.
the parent of the a2r4 is r4.
the X of the a2r4 is "549249.9589225953".
the Y of the a2r4 is "2537237.545605307".

[create a3r4]
a3r4 is a area. "An area (3) in r4".
the printed name of the a3r4 is "Room 4".
Understand "Area 3 in Room 4" as a3r4.
the parent of the a3r4 is r4.
the X of the a3r4 is "549119.5000000008".
the Y of the a3r4 is "2537392.5635736953".

[create a47r4]
a47r4 is a area. "An area (47) in r4".
the printed name of the a47r4 is "Room 4".
Understand "Area 47 in Room 4" as a47r4.
the parent of the a47r4 is r4.
the X of the a47r4 is "549563.0".
the Y of the a47r4 is "2537440.0".

[create a48r4]
a48r4 is a area. "An area (48) in r4".
the printed name of the a48r4 is "Room 4".
Understand "Area 48 in Room 4" as a48r4.
the parent of the a48r4 is r4.
the X of the a48r4 is "549089.0".
the Y of the a48r4 is "2537440.0".

[create a14r5]
a14r5 is a area. "An area (14) in r5".
the printed name of the a14r5 is "Room 5".
Understand "Area 14 in Room 5" as a14r5.
the parent of the a14r5 is r5.
the X of the a14r5 is "549417.5427520212".
the Y of the a14r5 is "2535345.720895571".

[create a19r5]
a19r5 is a area. "An area (19) in r5".
the printed name of the a19r5 is "Room 5".
Understand "Area 19 in Room 5" as a19r5.
the parent of the a19r5 is r5.
the X of the a19r5 is "549308.375".
the Y of the a19r5 is "2535238.2310668877".

[create a30r5]
a30r5 is a area. "An area (30) in r5".
the printed name of the a30r5 is "Room 5".
Understand "Area 30 in Room 5" as a30r5.
the parent of the a30r5 is r5.
the X of the a30r5 is "549526.625".
the Y of the a30r5 is "2535238.620689655".

[create a41r5]
a41r5 is a area. "An area (41) in r5".
the printed name of the a41r5 is "Room 5".
Understand "Area 41 in Room 5" as a41r5.
the parent of the a41r5 is r5.
the X of the a41r5 is "549272.0".
the Y of the a41r5 is "2535200.0".

[create a51r5]
a51r5 is a area. "An area (51) in r5".
the printed name of the a51r5 is "Room 5".
Understand "Area 51 in Room 5" as a51r5.
the parent of the a51r5 is r5.
the X of the a51r5 is "549563.0".
the Y of the a51r5 is "2535200.0".

[create a25r6]
a25r6 is a area. "An area (25) in r6".
the printed name of the a25r6 is "Room 6".
Understand "Area 25 in Room 6" as a25r6.
the parent of the a25r6 is r6.
the X of the a25r6 is "545290.2994279686".
the Y of the a25r6 is "2535342.112798646".

[create a26r6]
a26r6 is a area. "An area (26) in r6".
the printed name of the a26r6 is "Room 6".
Understand "Area 26 in Room 6" as a26r6.
the parent of the a26r6 is r6.
the X of the a26r6 is "546033.8207015462".
the Y of the a26r6 is "2536310.453316771".

[create a27r6]
a27r6 is a area. "An area (27) in r6".
the printed name of the a27r6 is "Room 6".
Understand "Area 27 in Room 6" as a27r6.
the parent of the a27r6 is r6.
the X of the a27r6 is "546746.7281516473".
the Y of the a27r6 is "2535347.941871378".

[create a28r6]
a28r6 is a area. "An area (28) in r6".
the printed name of the a28r6 is "Room 6".
Understand "Area 28 in Room 6" as a28r6.
the parent of the a28r6 is r6.
the X of the a28r6 is "545048.8999999999".
the Y of the a28r6 is "2535221.582983809".

[create a29r6]
a29r6 is a area. "An area (29) in r6".
the printed name of the a29r6 is "Room 6".
Understand "Area 29 in Room 6" as a29r6.
the parent of the a29r6 is r6.
the X of the a29r6 is "546994.0".
the Y of the a29r6 is "2535221.2526893676".

[create a33r6]
a33r6 is a area. "An area (33) in r6".
the printed name of the a33r6 is "Room 6".
Understand "Area 33 in Room 6" as a33r6.
the parent of the a33r6 is r6.
the X of the a33r6 is "545363.0".
the Y of the a33r6 is "2535200.0".

[create a34r6]
a34r6 is a area. "An area (34) in r6".
the printed name of the a34r6 is "Room 6".
Understand "Area 34 in Room 6" as a34r6.
the parent of the a34r6 is r6.
the X of the a34r6 is "545328.0999999999".
the Y of the a34r6 is "2535273.628215993".

[create a35r6]
a35r6 is a area. "An area (35) in r6".
the printed name of the a35r6 is "Room 6".
Understand "Area 35 in Room 6" as a35r6.
the parent of the a35r6 is r6.
the X of the a35r6 is "546670.0".
the Y of the a35r6 is "2535200.0".

[create a36r6]
a36r6 is a area. "An area (36) in r6".
the printed name of the a36r6 is "Room 6".
Understand "Area 36 in Room 6" as a36r6.
the parent of the a36r6 is r6.
the X of the a36r6 is "546706.0".
the Y of the a36r6 is "2535270.6362077417".

[create a49r6]
a49r6 is a area. "An area (49) in r6".
the printed name of the a49r6 is "Room 6".
Understand "Area 49 in Room 6" as a49r6.
the parent of the a49r6 is r6.
the X of the a49r6 is "545014.0".
the Y of the a49r6 is "2535200.0".

[create a50r6]
a50r6 is a area. "An area (50) in r6".
the printed name of the a50r6 is "Room 6".
Understand "Area 50 in Room 6" as a50r6.
the parent of the a50r6 is r6.
the X of the a50r6 is "547030.0".
the Y of the a50r6 is "2535200.0".

[create d0]
d0 is a door. "A door between a17r0 and a26r6".
d0 is south of a17r0 and north of a26r6.

[create d2]
d2 is a door. "A door between a6r2 and a21r3".
d2 is south of a6r2 and north of a21r3.

[create d4]
d4 is a door. "A door between a11r3 and a2r4".
d4 is north of a11r3 and south of a2r4.

[create d6]
d6 is a door. "A door between a12r0 and a13r1".
d6 is south of a12r0 and north of a13r1.

[create d8]
d8 is a door. "A door between a0r4 and a14r5".
d8 is south of a0r4 and north of a14r5.

[create d10]
d10 is a door. "A door between a15r1 and a9r2".
d10 is north of a15r1 and south of a9r2.

northeast of a0r4 is southwest of a1r4.

west of a0r4 is east of a2r4.

northwest of a2r4 is southeast of a3r4.

northeast of a4r0 is southwest of a5r0.

northeast of a6r2 is southwest of a7r2.

southeast of a8r2 is northwest of a9r2.

west of a6r2 is east of a9r2.

northwest of a4r0 is southeast of a10r0.

southeast of a16r0 is northwest of a17r0.

southwest of a12r0 is northeast of a17r0.

south of a4r0 is north of a12r0.

northwest of a16r0 is southeast of a18r0.

southwest of a14r5 is northeast of a19r5.

northeast of a20r3 is southwest of a21r3.

southeast of a11r3 is northwest of a22r3.

southwest of a13r1 is northeast of a23r1.

southeast of a15r1 is northwest of a24r1.

northeast of a25r6 is southwest of a26r6.

southeast of a26r6 is northwest of a27r6.

southwest of a25r6 is northeast of a28r6.

southeast of a27r6 is northwest of a29r6.

southeast of a14r5 is northwest of a30r5.

southwest of a31r0 is northeast of a32r0.

northwest of a33r6 is southeast of a34r6.

northeast of a35r6 is southwest of a36r6.

northeast of a5r0 is southwest of a37r0.

northeast of a7r2 is southwest of a38r2.

northwest of a8r2 is southeast of a39r2.

northwest of a10r0 is southeast of a40r0.

southwest of a19r5 is northeast of a41r5.

southwest of a20r3 is northeast of a42r3.

southeast of a22r3 is northwest of a43r3.

southwest of a23r1 is northeast of a44r1.

southeast of a24r1 is northwest of a45r1.

northwest of a18r0 is southeast of a46r0.

northeast of a1r4 is southwest of a47r4.

northwest of a3r4 is southeast of a48r4.

southwest of a28r6 is northeast of a49r6.

southeast of a29r6 is northwest of a50r6.

southeast of a30r5 is northwest of a51r5.

southeast of a25r6 is northwest of a34r6.

southwest of a27r6 is northeast of a36r6.

west of a11r3 is east of a21r3.

east of a13r1 is west of a15r1.

northeast of a16r0 is southwest of a32r0.

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
the player is in a51r5.

the orientation of the player is 6.
