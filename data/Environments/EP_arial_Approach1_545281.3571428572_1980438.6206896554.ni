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

[create a0r0]
a0r0 is a area. "An area (0) in r0".
the printed name of the a0r0 is "Room 0".
Understand "Area 0 in Room 0" as a0r0.
the parent of the a0r0 is r0.
the X of the a0r0 is "546462.5".
the Y of the a0r0 is "1982470.659119898".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "547593.2618692741".
the Y of the a1r0 is "1982462.4624381997".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "545403.9633579734".
the Y of the a2r0 is "1982479.0844623803".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "545281.3571428568".
the Y of the a3r0 is "1982601.3793103448".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "547845.6480764118".
the Y of the a6r0 is "1981752.763769299".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "545416.575887403".
the Y of the a7r0 is "1981552.0".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "546462.5".
the Y of the a8r0 is "1981552.0".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "546715.25".
the Y of the a9r0 is "1981648.0".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "547742.2303650749".
the Y of the a10r0 is "1982331.2864015328".

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "547996.0".
the Y of the a20r0 is "1981696.0".

[create a21r0]
a21r0 is a area. "An area (21) in r0".
the printed name of the a21r0 is "Room 0".
Understand "Area 21 in Room 0" as a21r0.
the parent of the a21r0 is r0.
the X of the a21r0 is "547898.0110044015".
the Y of the a21r0 is "1981732.125".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "547862.0".
the Y of the a22r0 is "1982435.0".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "547753.0919555".
the Y of the a23r0 is "1982340.0401490673".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "547643.0".
the Y of the a24r0 is "1982591.0".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "547594.1699790796".
the Y of the a25r0 is "1982464.5198744773".

[create a29r0]
a29r0 is a area. "An area (29) in r0".
the printed name of the a29r0 is "Room 0".
Understand "Area 29 in Room 0" as a29r0.
the parent of the a29r0 is r0.
the X of the a29r0 is "546751.0".
the Y of the a29r0 is "1981680.0".

[create a30r0]
a30r0 is a area. "An area (30) in r0".
the printed name of the a30r0 is "Room 0".
Understand "Area 30 in Room 0" as a30r0.
the parent of the a30r0 is r0.
the X of the a30r0 is "545243.0".
the Y of the a30r0 is "1982640.0".

[create a4r1]
a4r1 is a area. "An area (4) in r1".
the printed name of the a4r1 is "Room 1".
Understand "Area 4 in Room 1" as a4r1.
the parent of the a4r1 is r1.
the X of the a4r1 is "546564.9265734265".
the Y of the a4r1 is "1981488.0".

[create a5r1]
a5r1 is a area. "An area (5) in r1".
the printed name of the a5r1 is "Room 1".
Understand "Area 5 in Room 1" as a5r1.
the parent of the a5r1 is r1.
the X of the a5r1 is "547570.7448661654".
the Y of the a5r1 is "1981448.2093226735".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "546755.6376811592".
the Y of the a11r1 is "1980528.0".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "546849.5".
the Y of the a12r1 is "1980432.0".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "545281.3571428572".
the Y of the a13r1 is "1980438.6206896554".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "545403.9633579734".
the Y of the a14r1 is "1980560.9155376197".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "546462.5".
the Y of the a15r1 is "1980569.3481445312".

[create a16r1]
a16r1 is a area. "An area (16) in r1".
the printed name of the a16r1 is "Room 1".
Understand "Area 16 in Room 1" as a16r1.
the parent of the a16r1 is r1.
the X of the a16r1 is "546849.5".
the Y of the a16r1 is "1980624.0".

[create a17r1]
a17r1 is a area. "An area (17) in r1".
the printed name of the a17r1 is "Room 1".
Understand "Area 17 in Room 1" as a17r1.
the parent of the a17r1 is r1.
the X of the a17r1 is "546518.8024054982".
the Y of the a17r1 is "1981488.0".

[create a18r1]
a18r1 is a area. "An area (18) in r1".
the printed name of the a18r1 is "Room 1".
Understand "Area 18 in Room 1" as a18r1.
the parent of the a18r1 is r1.
the X of the a18r1 is "547639.0".
the Y of the a18r1 is "1981296.0".

[create a19r1]
a19r1 is a area. "An area (19) in r1".
the printed name of the a19r1 is "Room 1".
Understand "Area 19 in Room 1" as a19r1.
the parent of the a19r1 is r1.
the X of the a19r1 is "547600.0833333334".
the Y of the a19r1 is "1981382.0633035717".

[create a26r1]
a26r1 is a area. "An area (26) in r1".
the printed name of the a26r1 is "Room 1".
Understand "Area 26 in Room 1" as a26r1.
the parent of the a26r1 is r1.
the X of the a26r1 is "546884.0".
the Y of the a26r1 is "1980400.0".

[create a27r1]
a27r1 is a area. "An area (27) in r1".
the printed name of the a27r1 is "Room 1".
Understand "Area 27 in Room 1" as a27r1.
the parent of the a27r1 is r1.
the X of the a27r1 is "545243.0".
the Y of the a27r1 is "1980400.0".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "546884.0".
the Y of the a28r1 is "1980656.0".

[create d0]
d0 is a door. "A door between a6r0 and a5r1".
d0 is southwest of a6r0 and northeast of a5r1.

[create d1]
d1 is a door. "A door between a9r0 and a4r1".
d1 is southwest of a9r0 and northeast of a4r1.

[create d2]
d2 is a door. "A door between a7r0 and a14r1".
d2 is south of a7r0 and north of a14r1.

[create d3]
d3 is a door. "A door between a8r0 and a17r1".
d3 is southeast of a8r0 and northwest of a17r1.

east of a0r0 is west of a1r0.

west of a0r0 is east of a2r0.

northwest of a2r0 is southeast of a3r0.

east of a4r1 is west of a5r1.

east of a7r0 is west of a8r0.

south of a2r0 is north of a7r0.

south of a0r0 is north of a8r0.

north of a6r0 is south of a10r0.

southeast of a1r0 is northwest of a10r0.

southeast of a11r1 is northwest of a12r1.

northeast of a13r1 is southwest of a14r1.

west of a11r1 is east of a15r1.

east of a14r1 is west of a15r1.

northeast of a11r1 is southwest of a16r1.

north of a15r1 is south of a17r1.

northwest of a18r1 is southeast of a19r1.

west of a20r0 is east of a21r0.

southwest of a22r0 is northeast of a23r0.

south of a24r0 is north of a25r0.

southeast of a12r1 is northwest of a26r1.

southwest of a13r1 is northeast of a27r1.

northeast of a16r1 is southwest of a28r1.

northeast of a9r0 is southwest of a29r0.

northwest of a3r0 is southeast of a30r0.

east of a6r0 is west of a21r0.

northeast of a10r0 is southwest of a23r0.

northeast of a1r0 is southwest of a25r0.

southeast of a5r1 is northwest of a19r1.

west of a4r1 is east of a17r1.

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
the player is in a13r1.

the orientation of the player is 6.
