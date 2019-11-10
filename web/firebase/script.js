const db = firebase.firestore(); // accessable instance variable for Firebase Firestore
var currentUser; // the currently connected user

/**
 * Receives a file InputEvent event and forwards the received input picture to Firebase
 * and other parts of the program.
 * @param {InputEvent} inputEvent - the file event used to access the file information
 */
function handleFileSelect(inputEvent) {
  inputEvent.stopPropagation(); // prevents other nodes from receiving this event
  inputEvent.preventDefault(); // prevents default results when no action is taken
  pushFileToFirebase(inputEvent.target.files[0]);
}

/**
 * Pushes a single image file to the Firebase storage module's /images/... reference
 * @param {object} file - the image file to be pushed to Firebase
 */
function pushFileToFirebase(file) {
  // a storage reference is needed to specify exactly where the file is going
  // it can be thought of as a filepath with directories of where the file is
  // going to be stored
  var storageRef = firebase.storage().ref("images/" + file.name)

  // push the file to firebase storage
  storageRef.put(file).then(function (snapshot) {
    console.log("Uploaded file " + file.name + " with size of " + snapshot.totalBytes + " bytes.");

    // let's grab the download url...
    snapshot.ref.getDownloadURL().then(function (url) {
      // ...and attach that image url to the users' Firestore subcollections
      attachImageToUser(url);
    });
  }).catch(function (error) {
    console.error("Upload failed:", error);
  });
}

/**
 * Creates a Firebase database document with a field containing the Firebase
 * storage download URL for an image added by the user.
 * @param {string} src - Firebase Storage download URL for an image
 */
function attachImageToUser(src) {
  db.collection("users").doc(currentUser.email).collection("images").add({
    image: src
  }).then(function () {
    console.log("Document written with user ID: ", currentUser.email);
  }).catch(function (error) {
    console.error("Error adding document: ", error);
  });
}

/**
 * Writes an image to the viewable HTML page using a given downloaded source URL
 * from Firebase Storage
 * @param {string} src - Firebase Storage download URL for an image
 */
function addImageToGallery(src) {
  var galleryHTML = document.getElementById("gallery");
  var imgNode = document.createElement("img");
  imgNode.setAttribute("src", src);
  galleryHTML.appendChild(imgNode);
}

/**
 * Creates a new document under the users collection in Firestore for new users
 * @param {object} user 
 */
function pushNewUserToFirebase(user) {
  db.collection("users").doc(user.email).set({
    name: user.displayName
  }).then(function () {
    console.log("Document written with ID: ", user.email);
  }).catch(function (error) {
    console.error("Error adding document: ", error);
  });
}

/**
 * Pulls all the images assigned to a user from Firestore and calls for the images 
 * to be written to the viewable HTML page
 */
function displayGallery() {
  db.collection("users").doc(currentUser.email).collection("images").get().then(function (querySnapshot) {
    querySnapshot.forEach(function (doc) {
      // doc.data() is never undefined for query doc snapshots
      console.log(doc.id, " => ", doc.data());
      addImageToGallery(doc.data().image);
    });
  }).catch(function (error) {
    console.log("Error getting documents: ", error);
  });
}

/**
 * Checks to see if a given user has an document in the Firestore users collection and if no,
 * creates a new document for them
 * @param {object} user - User JS object retrieved from the Google login API
 */
function userExists(user) {
  var exists = db.collection("users").doc(user.email).get().then(function (doc) {
    if (!doc.exists) pushNewUserToFirebase(user);
  });
}

/**
 * Runs when the window is loaded to preform initial setup code such as event listeners.
 */
window.onload = function () {

  // add event listeners to the file selector and gallery buttons
  this.document.getElementById("file").addEventListener("change", handleFileSelect, false);
  this.document.getElementById("galleryBtn").addEventListener("click", displayGallery);

  var providerGoogle = new firebase.auth.GoogleAuthProvider(); // login with Google

  firebase.providerGoogle().useDeviceLanguage();
  firebase.providerGoogle().signInWithPopup(providerGoogle).then(function (result) {

    // this gives us a Google Access Token for accessing the Google API
    var token = result.credential.accessToken;

    // save the signed-in user info
    currentUser = result.user;

    // check to see if the user is already in our Firestore
    // if they are not, then we need to add them
    // we store userdata in Firestore to enable picture references
    // to be linked with a specific user
    userExists(currentUser);
  }).catch(function (error) {

    // handle errors here
    var errorCode = error.code;
    var errorMessage = error.message;

    // The email of the user's account used.
    var email = error.email;

    // The firebase.auth.AuthCredential type that was used.
    var credential = error.credential;
  });
}