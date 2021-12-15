/*!
* Start Bootstrap - Scrolling Nav v5.0.4 (https://startbootstrap.com/template/scrolling-nav)
* Copyright 2013-2021 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-scrolling-nav/blob/master/LICENSE)
*/
//
// Scripts
//


$(window).on('scroll load', function() {
    if ($(".navbar").offset().top > 20) {
        $(".fixed-top").addClass("top-nav-collapse");
    } else {
        $(".fixed-top").removeClass("top-nav-collapse");
    }
});

jQuery(document).ready(function ($) {
        $('#tabs').tab();
 });


var showImage = function(event) {
    var output2 = document.getElementById("recognized-image");
    output2.innerHTML = "";

    var output = document.getElementById("converted-image");
    output.src = URL.createObjectURL(event.target.files[0]);
    output.style.display = 'block';
    output.onload = function() {
      URL.revokeObjectURL(output.src) // free memory
    }
};

var showImage2 = function(event) {
    var output2 = document.getElementById("recognized-image");
    output2.innerHTML = "";

    var output = document.getElementById("converted-image2");
    output.src = URL.createObjectURL(event.target.files[0]);
    output.style.display = 'block';
    output.onload = function() {
      URL.revokeObjectURL(output.src) // free memory
    }
};

function showForm1() {
    document.getElementById("form-group2").style.display = 'none';
    document.getElementById("form-group1").style.display = 'block';
}

function showForm2() {
    document.getElementById("form-group1").style.display = 'none';
    document.getElementById("form-group2").style.display = 'block';
}

$( "#new-form" ).submit(function( event ) {

    // Stop form from submitting normally
    event.preventDefault();

    // Get some values from elements on the page:
    var $form = $(this);
    var option1 = $form.find("input[name='option1']").parent().hasClass("active");
    var option2 = $form.find("input[name='option2']").parent().hasClass("active");
    var text1 = $("#new-form #formControlSelect1").find(":selected").text();
    var text2 = $("#new-form input[name=exampleInput]").val();
    var url = $form.attr("action");


    // // Send the data using post
    // var posting = $.post(url, {
    //     "option1": option1, "option2": option2,
    //     "text1": text1, "text2": text2
    // });

    var formData = new FormData();
    var files = $('#picture1')[0].files;

    formData.append("option1", option1);
    formData.append("option2", option2);
    formData.append("text1", text1);
    formData.append("text2", text2);

    var selectedClass = text1;
    if (option2) {
        selectedClass = text2;
    }
    console.log(option1, option2, selectedClass);
    $( "#submit-image1").html( "<i class=\"fas fa-spinner fa-spin\"></i> Виконання запиту..." ).addClass( "disabled" );

    // Check file selected or not
    if (files.length > 0) {
        formData.append('image', files[0]);

        $.ajax({
            url: url,
            type: 'post',
            data: formData,
            contentType: false,
            processData: false,
            success: function(data){
                 $('#msg').after(`<div class="alert alert-success mt-2" >Клас <strong>${selectedClass}</strong> успішно додано!<button type="button" class="close" onclick="$(this).parent().remove();">&times;</button></div>`);
		    },
            error: function(data){
                $('#msg').after('<div class="alert alert-danger mt-2">Виникла помилка при додаванні класу!' +
                    '<button type="button" class="close" onclick="$(this).parent().remove();">&times;</button></div>');
            },
            complete: function(data) {
                $( "#submit-image1").html( "<i class=\"fas fa-plus\"></i>ДОДАТИ ЗОБРАЖЕННЯ" ).removeClass( "disabled" );
            }
        });
    }

});