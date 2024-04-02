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

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "525838.8491587283".
the Y of the a11r0 is "1383860.2680626006".

[create a15r0]
a15r0 is a area. "An area (15) in r0".
the printed name of the a15r0 is "Room 0".
Understand "Area 15 in Room 0" as a15r0.
the parent of the a15r0 is r0.
the X of the a15r0 is "525866.0".
the Y of the a15r0 is "1383880.0".

[create a0r3]
a0r3 is a area. "An area (0) in r3".
the printed name of the a0r3 is "Room 3".
Understand "Area 0 in Room 3" as a0r3.
the parent of the a0r3 is r3.
the X of the a0r3 is "527634.8727524203".
the Y of the a0r3 is "1385756.6882780083".

[create a1r3]
a1r3 is a area. "An area (1) in r3".
the printed name of the a1r3 is "Room 3".
Understand "Area 1 in Room 3" as a1r3.
the parent of the a1r3 is r3.
the X of the a1r3 is "528581.085465707".
the Y of the a1r3 is "1385559.569250254".

[create a4r3]
a4r3 is a area. "An area (4) in r3".
the printed name of the a4r3 is "Room 3".
Understand "Area 4 in Room 3" as a4r3.
the parent of the a4r3 is r3.
the X of the a4r3 is "527615.3784065933".
the Y of the a4r3 is "1384879.5".

[create a5r3]
a5r3 is a area. "An area (5) in r3".
the printed name of the a5r3 is "Room 3".
Understand "Area 5 in Room 3" as a5r3.
the parent of the a5r3 is r3.
the X of the a5r3 is "528312.6801725186".
the Y of the a5r3 is "1384927.2069072586".

[create a6r3]
a6r3 is a area. "An area (6) in r3".
the printed name of the a6r3 is "Room 3".
Understand "Area 6 in Room 3" as a6r3.
the parent of the a6r3 is r3.
the X of the a6r3 is "527614.7307586102".
the Y of the a6r3 is "1385740.3229080373".

[create a8r3]
a8r3 is a area. "An area (8) in r3".
the printed name of the a8r3 is "Room 3".
Understand "Area 8 in Room 3" as a8r3.
the parent of the a8r3 is r3.
the X of the a8r3 is "527228.9001965042".
the Y of the a8r3 is "1385593.9469272909".

[create a9r3]
a9r3 is a area. "An area (9) in r3".
the printed name of the a9r3 is "Room 3".
Understand "Area 9 in Room 3" as a9r3.
the parent of the a9r3 is r3.
the X of the a9r3 is "528645.2655038759".
the Y of the a9r3 is "1385587.1666666665".

[create a10r3]
a10r3 is a area. "An area (10) in r3".
the printed name of the a10r3 is "Room 3".
Understand "Area 10 in Room 3" as a10r3.
the parent of the a10r3 is r3.
the X of the a10r3 is "527650.005982906".
the Y of the a10r3 is "1385905.7505982905".

[create a13r3]
a13r3 is a area. "An area (13) in r3".
the printed name of the a13r3 is "Room 3".
Understand "Area 13 in Room 3" as a13r3.
the parent of the a13r3 is r3.
the X of the a13r3 is "528725.0".
the Y of the a13r3 is "1385622.0".

[create a14r3]
a14r3 is a area. "An area (14) in r3".
the printed name of the a14r3 is "Room 3".
Understand "Area 14 in Room 3" as a14r3.
the parent of the a14r3 is r3.
the X of the a14r3 is "527683.0".
the Y of the a14r3 is "1385904.0".

[create a2r4]
a2r4 is a area. "An area (2) in r4".
the printed name of the a2r4 is "Room 4".
Understand "Area 2 in Room 4" as a2r4.
the parent of the a2r4 is r4.
the X of the a2r4 is "527391.6749248416".
the Y of the a2r4 is "1384019.4923790898".

[create a3r4]
a3r4 is a area. "An area (3) in r4".
the printed name of the a3r4 is "Room 4".
Understand "Area 3 in Room 4" as a3r4.
the parent of the a3r4 is r4.
the X of the a3r4 is "527558.469293531".
the Y of the a3r4 is "1384019.674768928".

[create a7r4]
a7r4 is a area. "An area (7) in r4".
the printed name of the a7r4 is "Room 4".
Understand "Area 7 in Room 4" as a7r4.
the parent of the a7r4 is r4.
the X of the a7r4 is "526929.350578465".
the Y of the a7r4 is "1383860.99163803".

[create a12r4]
a12r4 is a area. "An area (12) in r4".
the printed name of the a12r4 is "Room 4".
Understand "Area 12 in Room 4" as a12r4.
the parent of the a12r4 is r4.
the X of the a12r4 is "526898.0".
the Y of the a12r4 is "1383880.0".

east of a0r3 is west of a1r3.

east of a2r4 is west of a3r4.

east of a4r3 is west of a5r3.

north of a4r3 is south of a6r3.

west of a2r4 is east of a7r4.

southwest of a1r3 is northeast of a5r3.

northeast of a1r3 is southwest of a9r3.

southwest of a0r3 is northeast of a6r3.

north of a0r3 is south of a10r3.

west of a6r3 is east of a8r3.

northwest of a7r4 is southeast of a12r4.

northeast of a9r3 is southwest of a13r3.

east of a10r3 is west of a14r3.

northeast of a11r0 is southwest of a15r0.

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
the player is in a13r3.

the orientation of the player is 6.
