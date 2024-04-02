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

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "545089.0".
the Y of the a10r0 is "1330053.0".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "545075.0".
the Y of the a11r0 is "1330114.4050772628".

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "545175.0".
the Y of the a12r0 is "1330053.0".

[create a13r0]
a13r0 is a area. "An area (13) in r0".
the printed name of the a13r0 is "Room 0".
Understand "Area 13 in Room 0" as a13r0.
the parent of the a13r0 is r0.
the X of the a13r0 is "545153.5".
the Y of the a13r0 is "1330138.00390625".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "546243.0".
the Y of the a14r0 is "1330563.0".

[create a15r0]
a15r0 is a area. "An area (15) in r0".
the printed name of the a15r0 is "Room 0".
Understand "Area 15 in Room 0" as a15r0.
the parent of the a15r0 is r0.
the X of the a15r0 is "546279.0".
the Y of the a15r0 is "1330658.9814757747".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "548415.0".
the Y of the a16r0 is "1330600.0".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "548378.875".
the Y of the a17r0 is "1330685.7209595963".

[create a18r0]
a18r0 is a area. "An area (18) in r0".
the printed name of the a18r0 is "Room 0".
Understand "Area 18 in Room 0" as a18r0.
the parent of the a18r0 is r0.
the X of the a18r0 is "549023.0".
the Y of the a18r0 is "1331227.0".

[create a19r0]
a19r0 is a area. "An area (19) in r0".
the printed name of the a19r0 is "Room 0".
Understand "Area 19 in Room 0" as a19r0.
the parent of the a19r0 is r0.
the X of the a19r0 is "548934.3368675595".
the Y of the a19r0 is "1331262.2857142857".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "544899.0".
the Y of the a22r0 is "1330277.0".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "544946.6865942029".
the Y of the a23r0 is "1330289.675111033".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "547558.0".
the Y of the a24r0 is "1331116.0".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "547524.3349056605".
the Y of the a25r0 is "1331096.278301887".

[create a34r0]
a34r0 is a area. "An area (34) in r0".
the printed name of the a34r0 is "Room 0".
Understand "Area 34 in Room 0" as a34r0.
the parent of the a34r0 is r0.
the X of the a34r0 is "546295.5248551389".
the Y of the a34r0 is "1330699.0127550354".

[create a35r0]
a35r0 is a area. "An area (35) in r0".
the printed name of the a35r0 is "Room 0".
Understand "Area 35 in Room 0" as a35r0.
the parent of the a35r0 is r0.
the X of the a35r0 is "548357.4184328839".
the Y of the a35r0 is "1330736.1969348688".

[create a36r0]
a36r0 is a area. "An area (36) in r0".
the printed name of the a36r0 is "Room 0".
Understand "Area 36 in Room 0" as a36r0.
the parent of the a36r0 is r0.
the X of the a36r0 is "548838.0324175665".
the Y of the a36r0 is "1331302.013516861".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "545107.9912057455".
the Y of the a40r0 is "1330164.2196130906".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "545141.5483636886".
the Y of the a41r0 is "1330166.9119265778".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "547338.8798686466".
the Y of the a44r0 is "1330769.695335895".

[create a45r0]
a45r0 is a area. "An area (45) in r0".
the printed name of the a45r0 is "Room 0".
Understand "Area 45 in Room 0" as a45r0.
the parent of the a45r0 is r0.
the X of the a45r0 is "547360.9702058504".
the Y of the a45r0 is "1331033.3234019503".

[create a47r0]
a47r0 is a area. "An area (47) in r0".
the printed name of the a47r0 is "Room 0".
Understand "Area 47 in Room 0" as a47r0.
the parent of the a47r0 is r0.
the X of the a47r0 is "545732.2363174232".
the Y of the a47r0 is "1331324.4781445055".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "545175.0".
the Y of the a0r1 is "1332624.0".

[create a1r1]
a1r1 is a area. "An area (1) in r1".
the printed name of the a1r1 is "Room 1".
Understand "Area 1 in Room 1" as a1r1.
the parent of the a1r1 is r1.
the X of the a1r1 is "545217.2698675498".
the Y of the a1r1 is "1332642.4980132452".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "546005.0".
the Y of the a2r1 is "1332624.0".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "545962.1798056157".
the Y of the a3r1 is "1332642.0199784017".

[create a4r1]
a4r1 is a area. "An area (4) in r1".
the printed name of the a4r1 is "Room 1".
Understand "Area 4 in Room 1" as a4r1.
the parent of the a4r1 is r1.
the X of the a4r1 is "545995.0".
the Y of the a4r1 is "1332714.0".

[create a5r1]
a5r1 is a area. "An area (5) in r1".
the printed name of the a5r1 is "Room 1".
Understand "Area 5 in Room 1" as a5r1.
the parent of the a5r1 is r1.
the X of the a5r1 is "545964.1050372064".
the Y of the a5r1 is "1332687.7894485786".

[create a6r1]
a6r1 is a area. "An area (6) in r1".
the printed name of the a6r1 is "Room 1".
Understand "Area 6 in Room 1" as a6r1.
the parent of the a6r1 is r1.
the X of the a6r1 is "545184.0".
the Y of the a6r1 is "1332714.0".

[create a7r1]
a7r1 is a area. "An area (7) in r1".
the printed name of the a7r1 is "Room 1".
Understand "Area 7 in Room 1" as a7r1.
the parent of the a7r1 is r1.
the X of the a7r1 is "545215.0876068376".
the Y of the a7r1 is "1332688.1662393163".

[create a8r1]
a8r1 is a area. "An area (8) in r1".
the printed name of the a8r1 is "Room 1".
Understand "Area 8 in Room 1" as a8r1.
the parent of the a8r1 is r1.
the X of the a8r1 is "546237.0".
the Y of the a8r1 is "1332732.0".

[create a9r1]
a9r1 is a area. "An area (9) in r1".
the printed name of the a9r1 is "Room 1".
Understand "Area 9 in Room 1" as a9r1.
the parent of the a9r1 is r1.
the X of the a9r1 is "546273.2142857142".
the Y of the a9r1 is "1332648.6123051257".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "548506.8043478262".
the Y of the a31r1 is "1332575.0".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "548470.1161675743".
the Y of the a32r1 is "1332583.577471612".

[create a33r1]
a33r1 is a area. "An area (33) in r1".
the printed name of the a33r1 is "Room 1".
Understand "Area 33 in Room 1" as a33r1.
the parent of the a33r1 is r1.
the X of the a33r1 is "546288.544564873".
the Y of the a33r1 is "1332614.554657809".

[create a38r1]
a38r1 is a area. "An area (38) in r1".
the printed name of the a38r1 is "Room 1".
Understand "Area 38 in Room 1" as a38r1.
the parent of the a38r1 is r1.
the X of the a38r1 is "545232.1884979304".
the Y of the a38r1 is "1332659.7787601028".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "545947.1865386876".
the Y of the a39r1 is "1332659.6370670423".

[create a42r1]
a42r1 is a area. "An area (42) in r1".
the printed name of the a42r1 is "Room 1".
Understand "Area 42 in Room 1" as a42r1.
the parent of the a42r1 is r1.
the X of the a42r1 is "547338.698041558".
the Y of the a42r1 is "1332533.2713725318".

[create a43r1]
a43r1 is a area. "An area (43) in r1".
the printed name of the a43r1 is "Room 1".
Understand "Area 43 in Room 1" as a43r1.
the parent of the a43r1 is r1.
the X of the a43r1 is "545585.3841463415".
the Y of the a43r1 is "1332539.4588414633".

[create a46r1]
a46r1 is a area. "An area (46) in r1".
the printed name of the a46r1 is "Room 1".
Understand "Area 46 in Room 1" as a46r1.
the parent of the a46r1 is r1.
the X of the a46r1 is "545713.0653848067".
the Y of the a46r1 is "1331917.7455625804".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "548700.0".
the Y of the a26r2 is "1332475.677419355".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "548722.3654078756".
the Y of the a27r2 is "1332438.9653042448".

[create a20r3]
a20r3 is a area. "An area (20) in r3".
the printed name of the a20r3 is "Room 3".
Understand "Area 20 in Room 3" as a20r3.
the parent of the a20r3 is r3.
the X of the a20r3 is "548700.0".
the Y of the a20r3 is "1332575.0".

[create a21r3]
a21r3 is a area. "An area (21) in r3".
the printed name of the a21r3 is "Room 3".
Understand "Area 21 in Room 3" as a21r3.
the parent of the a21r3 is r3.
the X of the a21r3 is "548667.8007246377".
the Y of the a21r3 is "1332550.1693548388".

[create a26r3]
a26r3 is a area. "An area (26) in r3".
the printed name of the a26r3 is "Room 3".
Understand "Area 26 in Room 3" as a26r3.
the parent of the a26r3 is r3.
the X of the a26r3 is "548700.0".
the Y of the a26r3 is "1332475.677419355".

[create a28r3]
a28r3 is a area. "An area (28) in r3".
the printed name of the a28r3 is "Room 3".
Understand "Area 28 in Room 3" as a28r3.
the parent of the a28r3 is r3.
the X of the a28r3 is "548684.6793707323".
the Y of the a28r3 is "1332500.825751474".

[create a29r3]
a29r3 is a area. "An area (29) in r3".
the printed name of the a29r3 is "Room 3".
Understand "Area 29 in Room 3" as a29r3.
the parent of the a29r3 is r3.
the X of the a29r3 is "548525.0".
the Y of the a29r3 is "1332566.0".

[create a30r3]
a30r3 is a area. "An area (30) in r3".
the printed name of the a30r3 is "Room 3".
Understand "Area 30 in Room 3" as a30r3.
the parent of the a30r3 is r3.
the X of the a30r3 is "548603.4021739131".
the Y of the a30r3 is "1332563.552540156".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "548506.8043478262".
the Y of the a31r3 is "1332575.0".

[create a37r3]
a37r3 is a area. "An area (37) in r3".
the printed name of the a37r3 is "Room 3".
Understand "Area 37 in Room 3" as a37r3.
the parent of the a37r3 is r3.
the X of the a37r3 is "548661.8286071355".
the Y of the a37r3 is "1332542.424978698".

[create d0]
d0 is a door. "A door between a45r0 and a42r1".
d0 is north of a45r0 and south of a42r1.

[create d1]
d1 is a door. "A door between a47r0 and a46r1".
d1 is north of a47r0 and south of a46r1.

[create d2]
d2 is a door. "A door between a36r0 and a27r2".
d2 is north of a36r0 and south of a27r2.

northeast of a0r1 is southwest of a1r1.

northwest of a2r1 is southeast of a3r1.

southwest of a4r1 is northeast of a5r1.

southeast of a6r1 is northwest of a7r1.

southeast of a8r1 is northwest of a9r1.

north of a10r0 is south of a11r0.

north of a12r0 is south of a13r0.

north of a14r0 is south of a15r0.

northwest of a16r0 is southeast of a17r0.

west of a18r0 is east of a19r0.

southwest of a20r3 is northeast of a21r3.

east of a22r0 is west of a23r0.

southwest of a24r0 is northeast of a25r0.

southeast of a26r2 is northwest of a27r2.

east of a26r2 is west of a26r2.

east of a29r3 is west of a30r3.

west of a31r1 is east of a32r1.

southeast of a9r1 is northwest of a33r1.

north of a15r0 is south of a34r0.

northwest of a17r0 is southeast of a35r0.

west of a19r0 is east of a36r0.

southwest of a21r3 is northeast of a37r3.

northeast of a1r1 is southwest of a38r1.

northwest of a3r1 is southeast of a39r1.

southwest of a5r1 is northeast of a39r1.

southeast of a7r1 is northwest of a38r1.

northeast of a11r0 is southwest of a40r0.

north of a13r0 is south of a41r0.

east of a30r3 is west of a37r3.

west of a32r1 is east of a42r1.

west of a39r1 is east of a43r1.

east of a33r1 is west of a42r1.

east of a38r1 is west of a43r1.

north of a44r0 is south of a45r0.

west of a25r0 is east of a45r0.

south of a43r1 is north of a46r1.

southwest of a33r1 is northeast of a46r1.

northwest of a28r3 is southeast of a37r3.

east of a40r0 is west of a41r0.

northeast of a41r0 is southwest of a47r0.

southeast of a23r0 is northwest of a40r0.

east of a34r0 is west of a44r0.

west of a35r0 is east of a44r0.

northwest of a34r0 is southeast of a47r0.

northeast of a35r0 is southwest of a36r0.

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
the player is in a25r0.

the orientation of the player is 6.
