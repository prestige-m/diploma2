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
    var output = document.getElementById("converted-image");
    output.src = URL.createObjectURL(event.target.files[0]);
    output.style.display = 'block';
    output.onload = function() {
      URL.revokeObjectURL(output.src) // free memory
    }
};

var showImage2 = function(event) {
    var output = document.getElementById("converted-image2");
    output.src = URL.createObjectURL(event.target.files[0]);
    output.style.display = 'block';
    output.onload = function() {
      URL.revokeObjectURL(output.src) // free memory
    }
};


/*
window.addEventListener('DOMContentLoaded', event => {

    // Activate Bootstrap scrollspy on the main nav element
    const mainNav = document.body.querySelector('#mainNav');
    if (mainNav) {
        new bootstrap.ScrollSpy(document.body, {
            target: '#mainNav',
            offset: 74,
        });
    };

    // Collapse responsive navbar when toggler is visible
    const navbarToggler = document.body.querySelector('.navbar-toggler');
    const responsiveNavItems = [].slice.call(
        document.querySelectorAll('#navbarResponsive .nav-link')
    );
    responsiveNavItems.map(function (responsiveNavItem) {
        responsiveNavItem.addEventListener('click', () => {
            if (window.getComputedStyle(navbarToggler).display !== 'none') {
                navbarToggler.click();
            }
        });
    });

});

*/

